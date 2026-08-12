import copy
import gzip
import json
import logging
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

import bpy
import msgpack
import mujoco as mj
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.housegen.constants import (
    ARTICULABLE_DYNAMIC_CATEGORIES,
    ARTICULABLE_DYNAMIC_CLASS,
    DEFAULT_SETTLE_TIME,
    DYNAMIC_CLASS,
    DYNAMIC_OBJ_GEOMS_MARGIN,
    FREE_JOINT_DAMPING,
    FREE_JOINT_FRICTIONLOSS,
    ITHOR_HOUSES_FLOOR_OFFSET,
    STRUCTURAL_CLASS,
    STRUCTURAL_WALL_CLASS,
    TYPES_TO_REMOVE_ALL_JOINTS,
    VALID_DYNAMIC_CLASSES,
    VALID_VISUAL_CLASSES,
    VISUAL_CLASS,
)
from molmo_spaces.housegen.utils import (
    NameMapping,
    ObjectNameMapping,
    SceneInfo,
    SceneObjectInfo,
    SceneObjectType,
    change_name_recursively,
    create_room_mesh,
    create_wall_mesh,
    generate_body_name,
    generate_object_hash,
    get_free_bodies_ids,
    get_root_bodies_handles,
    is_frozen_in_space,
    make_hole_in_wall,
    make_wall_colliders,
    must_use_prim,
    set_defaults,
    should_flip_asset_direction,
    stdout_redirected,
    unity_to_mj_pos,
)
from molmo_spaces.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA as THOR_TYPE_TO_LEMMA,
)

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

HOUSE_BASE_XML = Path(__file__).parent.parent / "resources" / "base_scene.xml"

DEFAULT_COLOR = np.array([0.5, 0.5, 0.5, 1.0])
DEFAULT_WALL_COLOR = np.array([0.1, 0.8, 0.1, 1.0])
COLOR_NONE = np.array([0.0, 0.0, 0.0, 0.0])
DEFAULT_MESH_SCALE = np.array([-1, 1, 1], dtype=np.float64)
DEFAULT_SITES_GROUP = 5
DEFAULT_MIN_WALL_THICKNESS = 0.001

THRESHOLD_TRANSPARENCY = 1e-7

ITHOR_CUSTOM_OBJ_CATEGORIES = [
    "Cabinet",
    "Drawer",
    "ShowerDoor",
    "Oven",
    "Dishwasher",
    "StoveKnob",
]

# These assets are problematic, so for now we're just skipping them and their children
THOR_ASSETS_TO_SKIP = {"keychain", "watch", "creditcard"}

# These assets give errors when loaded by themselves, so for now just skip them
THOR_ASSETS_INVALID_MUJOCO = {
    "RoboTHOR_dresser_aneboda",  # min-inertia issues
    "Laptop_20",  # collider with volume too small
    "bin_11",  # collider with volume too small
    "bin_6",  # collider with volume too small
}

# This specific fork gives errors when used, so for now replace it with a regular fork
THOR_ASSETS_TO_SWAP: dict[str, str] = {
    "RoboTHOR_fork_ai2_v": "Fork_1",
}

FLOOR_Y_OFFSET = 0.01
FRIDGE_Z_OFFSET = 0.025

# Compensation transform used bc of a mismatch from the exported assets from Unity
T_TRANSFORM = np.eye(4)
T_TRANSFORM[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
T_TRANSFORM[:3, 3] = [0, 0, 0]

# Extra rotation used bc of a mismatch from the exported assets from Unity
T_FIX_ROTMAT = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()

REFERENCE_EXTENT = 24.552953504964606  # extent of val_1 scene
REF_SCALE_CONTACT_WIDTH = 0.025
REF_SCALE_CONTACT_HEIGHT = 0.025
REF_SCALE_JOINT_LENGTH = 0.2
REF_SCALE_JOINT_WIDTH = 0.025
REF_SCALE_FRAME_LENGTH = 0.15
REF_SCALE_FRAME_WIDTH = 0.0125

INTERSECTION_THRESHOLD = -0.01


class SceneType(str, Enum):
    ITHOR = "ithor"
    PROCTHOR_10K = "procthor-10k"
    PROCTHOR_OBJAVERSE = "procthor-objaverse"
    HOLODECK = "holodeck-objaverse"


class LogDict(TypedDict):
    warnings: list[str]
    errors: list[str]


@dataclass
class ExportFlags:
    export_walls: bool = True
    export_rooms: bool = True
    export_doors: bool = True
    export_windows: bool = True
    export_thor_objects: bool = True
    export_objaverse: bool = True
    export_custom_ithor: bool = True
    export_ceiling: bool = False
    export_lights: bool = False


@dataclass
class RoomSettings:
    generate_uvs: bool = True
    uv_scale_factor: float = 5.0


@dataclass
class SceneBuildConfig:
    # The type of scene to be built
    scene_type: SceneType
    # The path to the root assets directory (where the ThorAssets, scenes, etc. are stored)
    assets_dir: Path
    # The path to where the converted mjcf versions of the objaverse assets are stored
    objaverse_mjcf_dir: Path | None = None
    # The path to the original data for the objaverse assets (contains the msgpack vertex data)
    objaverse_data_dir: Path | None = None
    # The path where to save the generated house
    target_dir: Path | None = None
    # The string version of the relative path to the assets dir for the generated scene
    target_assets_dir_str: str | None = None
    # The path where all generated assets (.png, .obj, etc.) are stored
    target_assets_dir: Path | None = None
    # Whether or not should overwrite an existing mesh .obj
    overwrite_mesh: bool = True
    # Flag to indicate if we're going to export objects mesh colliders when possible
    use_mesh_colliders: bool = True
    # Configuration options to use when exporting
    flags: ExportFlags = field(default_factory=ExportFlags)
    # Info about how populated is the scene
    counts: defaultdict[SceneObjectType, int] = field(default_factory=lambda: defaultdict(int))
    # The id of the house to be built
    house_id: str | None = None
    # Settings used when exporting the rooms
    room_opts: RoomSettings = field(default_factory=RoomSettings)
    # Use sleeping islands feature (requires mujoco>=3.3.8)
    use_sleep_island: bool = False
    # Settle time to use when generating the house
    settle_time: float = DEFAULT_SETTLE_TIME
    # Whether or not to make a copy of the scene previous to the settle stage
    copy_non_settled: bool = False
    # Whether or not to make a copy of the original file (before applying filters from tests)
    copy_original: bool = False
    # Whether or not to remove sites when adding a mjcf model (required for USD conversion for now)
    remove_sites: bool = False
    # Whether or not to run a filter pass for holodeck houses to remove duplicates
    filter_stage_holodeck_duplicates: bool = True
    # Whether or not to run a filter pass for holodeck houses to remove objects in deep intersection
    filter_stage_holodeck_intersection: bool = True
    # Parameters used for stability ------------------------------
    param_geom_margin: float = DYNAMIC_OBJ_GEOMS_MARGIN
    param_freejoint_damping: float = FREE_JOINT_DAMPING
    param_freejoint_frictionloss: float = FREE_JOINT_FRICTIONLOSS
    # ------------------------------------------------------------


@dataclass
class Texture2dInfo:
    name: str
    file: Path
    builtin: str = "none"


@dataclass
class MaterialInfo:
    name: str
    rgba: np.ndarray | None = None
    texture: Texture2dInfo | None = None
    specular: float = 0.5
    shininess: float = 0.5


@dataclass
class MeshInfo:
    name: str
    file: Path
    scale: np.ndarray
    inertia: mj.mjtMeshInertia = mj.mjtMeshInertia.mjMESH_INERTIA_LEGACY


@dataclass
class ModelInfo:
    prim_path: Path | None = None
    mesh_path: Path | None = None
    fallback_path: Path | None = None

    def has_any_model(self) -> bool:
        return (
            (self.prim_path is not None)
            or (self.mesh_path is not None)
            or (self.fallback_path is not None)
        )


@dataclass
class AssetsCache:
    textures: dict[str, Texture2dInfo] = field(default_factory=dict)
    materials: dict[str, MaterialInfo] = field(default_factory=dict)
    meshes: dict[str, MeshInfo] = field(default_factory=dict)


@dataclass
class WallCachedInfo:
    filepath: Path
    vertices: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    geom_name: str
    visual_name: str
    body_name: str
    room_id: int = 0
    mat_info: MaterialInfo | None = None
    mesh_colliders_paths: list[Path] = field(default_factory=list)


@dataclass
class RoomCachedInfo:
    filepath: Path
    mesh: o3d.geometry.TriangleMesh
    vertices: np.ndarray


class MlSpacesSceneBuilder:
    def __init__(
        self,
        scene_type: SceneType,
        asset_dir: Path,
        asset_id_to_object_type: dict[str, str],
        materials_to_textures: dict[str, Any],
        objaverse_mjcf_dir: Path | None = None,
        objaverse_data_dir: Path | None = None,
        ignore_door_kinematic_flag: bool = True,
        ignore_kinematic_flag: bool = False,
        use_mesh_colliders: bool = True,
        use_mesh_collider_for_objects_on: list | None = None,  # ["surface"],
        add_ceiling: bool = False,
        export_lights: bool = False,
        use_sleep_island: bool = False,
        settle_time: float = DEFAULT_SETTLE_TIME,
        copy_original: bool = False,
        copy_non_settled: bool = False,
        remove_sites: bool = False,
    ):
        self.cfg: SceneBuildConfig = SceneBuildConfig(
            scene_type=scene_type,
            assets_dir=asset_dir,
            objaverse_mjcf_dir=objaverse_mjcf_dir,
            objaverse_data_dir=objaverse_data_dir,
            use_mesh_colliders=use_mesh_colliders,
            use_sleep_island=use_sleep_island,
            settle_time=settle_time,
            copy_original=copy_original,
            copy_non_settled=copy_non_settled,
            remove_sites=remove_sites,
        )
        self.cfg.flags.export_ceiling = add_ceiling
        self.cfg.flags.export_lights = export_lights

        self.asset_id_to_object_type: dict[str, str] = asset_id_to_object_type

        self._room_cache: dict[str, RoomCachedInfo] = {}
        self._wall_cache: dict[str, WallCachedInfo] = {}
        self._mesh_cache: dict[str, o3d.geometry.TriangleMesh] = {}

        self._assets_cache: AssetsCache = AssetsCache()

        self._logs: LogDict = {"warnings": [], "errors": []}

        self._thor_materials: dict[str, Any] = materials_to_textures

        self._cache_exclusion_pairs: set[str] = set()
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._scene_info: SceneInfo = SceneInfo()

        self._textures_added_set: set[str] = set()
        self._materials_added_set: set[str] = set()

        self._name_mapping: NameMapping = NameMapping()

        # Cache the path to all THOR assets in the assets folder
        self._cache_id_to_path: defaultdict[str, ModelInfo] = defaultdict(lambda: ModelInfo())
        thor_folder_path = self.cfg.assets_dir / "objects" / "thor"
        for candidate_xml in thor_folder_path.rglob("*.xml"):
            if "_old" in candidate_xml.stem.lower():
                continue
            if "_prim.xml" in candidate_xml.name.lower():
                key_id = candidate_xml.name.replace("_prim.xml", "").lower()
                self._cache_id_to_path[key_id].prim_path = candidate_xml
            elif "_mesh.xml" in candidate_xml.name.lower():
                key_id = candidate_xml.name.replace("_mesh.xml", "").lower()
                self._cache_id_to_path[key_id].mesh_path = candidate_xml
            else:
                key_id = candidate_xml.stem.lower()
                self._cache_id_to_path[key_id].fallback_path = candidate_xml

        self.spec: mj.MjSpec | None = None
        self.spec_opts: mj.MjSpec | None = None

        # House Properties
        self.ignore_kinematic_flag = ignore_kinematic_flag
        self.ignore_door_kinematic_flag = ignore_door_kinematic_flag
        self.use_mesh_collider_for_objects_on = use_mesh_collider_for_objects_on

    @property
    def logs(self) -> LogDict:
        return self._logs

    def _parse_thor_materials(self) -> None:
        assert self.cfg.target_dir is not None, (
            "Must call this after setting target_dir when loading a house"
        )
        for mat_id, mat_json in self._thor_materials.items():
            mat_info = MaterialInfo(name=mat_id)
            if "albedo_rgba" in mat_json:
                rgba = np.array(list(map(float, mat_json["albedo_rgba"].split(" "))))
                # There are some materials that have plain black color with alpha=0. For these
                # cases we're assumming that the color is going to come from the material info
                # for that specific object in the house json. Otherwise, just fallback to a default
                if not np.allclose(rgba, COLOR_NONE, atol=1e-5):
                    mat_info.rgba = rgba
            # Be careful, as the json data contains some keys that are defined, but their values
            # are set to None, like the albedo texture
            if "specular" in mat_json and mat_json["specular"] is not None:
                mat_info.specular = float(mat_json["specular"])
            if "_MainTex" in mat_json and mat_json["_MainTex"] is not None:
                tex_unity_path = Path(mat_json["_MainTex"])
                tex_name = tex_unity_path.stem
                tex_filepath = (
                    self.cfg.assets_dir
                    / "objects"
                    / "thor"
                    / "Textures"
                    / f"{tex_unity_path.stem}.png"
                )
                if not tex_filepath.is_file():
                    msg = f"Texture '{tex_filepath.as_posix()}' doesn't exist"
                    self._logs["warnings"].append(msg)
                    log.warning(msg)
                relpath_to_target_dir = Path(
                    os.path.relpath(tex_filepath.parent, start=self.cfg.target_dir)
                )
                mat_info.texture = Texture2dInfo(
                    name=tex_name, file=relpath_to_target_dir / tex_filepath.name
                )

            self._assets_cache.materials[mat_id] = mat_info

    def load_from_json(
        self,
        thor_house: dict[str, Any],
        target_dir: Path,
        house_id: str,
        stability_params: dict[str, float] = {},
    ) -> mj.MjSpec | None:
        self.cfg.target_dir = target_dir
        self.cfg.target_dir.mkdir(exist_ok=True)
        self.cfg.house_id = house_id

        self._parse_thor_materials()

        def get_assets_dir_str() -> str:
            match self.cfg.scene_type:
                case SceneType.ITHOR:
                    return house_id
                case SceneType.PROCTHOR_10K | SceneType.PROCTHOR_OBJAVERSE:
                    return f"{house_id}_assets"
                case _:
                    return f"{house_id}_assets"

        self.cfg.target_assets_dir_str = get_assets_dir_str()
        self.cfg.target_assets_dir = self.cfg.target_dir / self.cfg.target_assets_dir_str
        self.cfg.target_assets_dir.mkdir(exist_ok=True)

        self.cfg.param_geom_margin = stability_params.get(
            "param_geom_margin", DYNAMIC_OBJ_GEOMS_MARGIN
        )
        self.cfg.param_freejoint_damping = stability_params.get(
            "param_freejoint_damping", FREE_JOINT_DAMPING
        )
        self.cfg.param_freejoint_frictionloss = stability_params.get(
            "param_freejoint_frictionloss", FREE_JOINT_FRICTIONLOSS
        )

        # Have to load from string, bc if loading from file the path will be set to the folder
        # location of that file, and that messes everyting up
        with open(HOUSE_BASE_XML, "r") as fhandle:
            xml_model_str = fhandle.read()
            self.spec = mj.MjSpec.from_string(xml_model_str)
            self.spec_opts = mj.MjSpec.from_string(xml_model_str)
        self.spec.modelname = house_id

        if self.cfg.use_mesh_colliders:
            self.spec.njmax = 5000
            self.spec.nconmax = 5000

        if self.cfg.use_sleep_island:
            self.spec.option.enableflags |= mj.mjtEnableBit.mjENBL_SLEEP

        main_def = self.spec.find_default("main")

        proc_parameters = thor_house.get("proceduralParameters", {})
        lights_data = proc_parameters.get("lights", [])
        has_any_lights = False
        if self.cfg.flags.export_lights:
            if len(lights_data) > 0:
                for light_json in lights_data:
                    if light_json.get("type", "").lower() == "directional":
                        continue

                    self.add_light(light_json)
                    has_any_lights = True

        if not has_any_lights:
            self.spec.worldbody.add_light(
                pos=[1, -1, 1.5],
                dir=[-1, 1, -1],
                diffuse=[0.5, 0.5, 0.5],
                type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
            )

        floor_geom = self.spec.worldbody.add_geom(
            name="floor",
            type=mj.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, FLOOR_Y_OFFSET],
        )
        set_defaults(floor_geom, self.spec.find_default(STRUCTURAL_CLASS), main_def)
        if self.cfg.scene_type == SceneType.ITHOR and house_id in ITHOR_HOUSES_FLOOR_OFFSET:
            floor_geom.pos[2] = ITHOR_HOUSES_FLOOR_OFFSET[house_id]

        rooms_data = thor_house.get("rooms", [])
        if self.cfg.flags.export_rooms:
            for room_json in rooms_data:
                self.add_room(room_json)

        walls_data = thor_house.get("walls", [])
        walls_data = [w for w in walls_data if not w.get("empty", False)]
        if self.cfg.flags.export_walls:
            for wall_json in walls_data:
                self.add_wall(wall_json)

        doors_data = thor_house.get("doors", [])
        if self.cfg.flags.export_doors:
            for door_json in doors_data:
                self.add_door(door_json)

        windows_data = thor_house.get("windows", [])
        if self.cfg.flags.export_windows:
            for window_json in windows_data:
                self.add_window(window_json)

        # Add UVs to walls with holes after adding doors and windows
        self.add_walls_uv([w["id"] for w in walls_data])

        objects_data = thor_house.get("objects", [])
        for object_json in objects_data:
            self.add_object_from_mjcf_model(object_json)

        structural_objects_data = thor_house.get("structuralObjects", [])
        for object_json in structural_objects_data:
            self.add_object_from_mjcf_model(object_json)

        proc_parameters = thor_house.get("proceduralParameters", {})
        if "skyboxId" in proc_parameters and self.cfg.scene_type != SceneType.ITHOR:
            skybox_id = proc_parameters["skyboxId"]
            skyboxes_folder = self.cfg.assets_dir / "objects" / "thor" / "SkyBox"
            texture_path = skyboxes_folder / f"{skybox_id}.png"
            if not texture_path.is_file():
                # Try find jpg and convert it, otherwise just use a random skybox
                jpg_texture_path = texture_path.parent / f"{texture_path.stem}.jpg"
                if not jpg_texture_path.is_file():
                    candidates = list(skyboxes_folder.glob("*.png"))
                    texture_path = random.choice(candidates)
                else:
                    pil_image = Image.open(jpg_texture_path)
                    pil_image.save(texture_path)
            relpath_to_target_dir = Path(
                os.path.relpath(texture_path.parent, start=self.cfg.target_dir)
            )
            self.spec.add_texture(
                name=texture_path.stem,
                type=mj.mjtTexture.mjTEXTURE_SKYBOX,
                file=(relpath_to_target_dir / texture_path.name).as_posix(),
                gridsize=(2, 4),
                # gridlayout=[v for v in "LFRB.D.."], # mjspec has an issue here, have to use XML
            )

        self.spec.compiler.balanceinertia = True

        if self.cfg.scene_type == SceneType.HOLODECK and self.cfg.filter_stage_holodeck_duplicates:
            root_bodies_handles: list[mj.MjsBody] = get_root_bodies_handles(self.spec)
            bodies_to_delete: dict[str, mj.MjsBody] = dict()
            for i in range(len(root_bodies_handles)):
                root_i = root_bodies_handles[i]
                for j in range(i + 1, len(root_bodies_handles)):
                    root_j = root_bodies_handles[j]
                    if np.allclose(root_i.pos, root_j.pos, atol=1e-4):
                        if root_i.name not in bodies_to_delete:  # Delete the first one for now
                            bodies_to_delete[root_i.name] = root_i

            for body_handle in bodies_to_delete.values():
                if body_handle is None:
                    continue
                self.spec.delete(body_handle)

        workdir = Path.cwd()
        os.chdir(self.cfg.target_dir)
        model: mj.MjModel = self.spec.compile()
        data: mj.MjData = mj.MjData(model)

        scale_vis = REFERENCE_EXTENT / model.stat.extent

        self.spec.visual.scale.contactheight = scale_vis * REF_SCALE_CONTACT_HEIGHT
        self.spec.visual.scale.contactwidth = scale_vis * REF_SCALE_CONTACT_WIDTH
        self.spec.visual.scale.jointlength = scale_vis * REF_SCALE_JOINT_LENGTH
        self.spec.visual.scale.jointwidth = scale_vis * REF_SCALE_JOINT_WIDTH
        self.spec.visual.scale.framelength = scale_vis * REF_SCALE_FRAME_LENGTH
        self.spec.visual.scale.framewidth = scale_vis * REF_SCALE_FRAME_WIDTH

        if (
            self.cfg.scene_type == SceneType.HOLODECK
            and self.cfg.filter_stage_holodeck_intersection
        ):
            freebodies_set = set(get_free_bodies_ids(model))
            bodies_ids_to_delete: set[int] = set()
            pairs_registered: set[int] = set()

            mj.mj_forward(model, data)
            for i_con in range(data.ncon):
                contact = data.contact[i_con]
                geom_id_1, geom_id_2 = contact.geom[0], contact.geom[1]
                body_id_1, body_id_2 = model.geom_bodyid[[geom_id_1, geom_id_2]]
                root_id_1, root_id_2 = model.body_rootid[[body_id_1, body_id_2]]

                if root_id_1 not in freebodies_set or root_id_2 not in freebodies_set:
                    continue

                if contact.dist >= INTERSECTION_THRESHOLD:
                    continue

                key = root_id_1 * root_id_2
                if key not in pairs_registered:
                    pairs_registered.add(key)
                    # For now, just delete the first body
                    bodies_ids_to_delete.add(root_id_1)

            if self.cfg.filter_stage_holodeck_duplicates:
                for body_id_a in freebodies_set:
                    for body_id_b in freebodies_set:
                        if body_id_a == body_id_b:
                            continue
                        if np.allclose(data.xpos[body_id_a], data.xpos[body_id_b], atol=1e-5):
                            key = body_id_a * body_id_b
                            if key not in pairs_registered:
                                pairs_registered.add(key)
                                bodies_ids_to_delete.add(body_id_a)

            for body_id in bodies_ids_to_delete:
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_id)
                body_handle = self.spec.body(body_name)
                if body_handle is not None:
                    self.spec.delete(body_handle)

        save_path = self.cfg.target_dir / f"{house_id}.xml"
        with open(save_path, "w") as fhandle:
            fhandle.write(self.spec.to_xml())
        self._apply_patches_to_mjcf_house(save_path)

        # Add the ceiling to the spec and export again to a separate file
        if self.cfg.flags.export_ceiling and self.cfg.scene_type != SceneType.ITHOR:
            room_z = 0.0
            for wall in thor_house["walls"][0]["polygon"]:
                room_z = max(room_z, wall["y"])
            for room_json in rooms_data:
                self.add_room(
                    room_json,
                    room_z=room_z,
                    name_prefix="ceiling",
                    material_name=thor_house["proceduralParameters"]["ceilingMaterial"]["name"],
                )

            _ = self.spec.compile()

            save_path = self.cfg.target_dir / f"{house_id}_ceiling.xml"
            with open(save_path, "w") as fhandle:
                fhandle.write(self.spec.to_xml())
            self._apply_patches_to_mjcf_house(save_path)

        os.chdir(workdir)
        # ------------------------------------------------------------------------------------------

        if self.cfg.copy_non_settled:
            original_path = self.cfg.target_dir / f"{house_id}.xml"
            copy_non_settled_path = self.cfg.target_dir / f"{house_id}_non_settled.xml"
            if original_path.is_file():
                shutil.copy(original_path, copy_non_settled_path)

        # Settle -----------------------------------------------------------------------------------
        self._process_settle(self.cfg.target_dir / f"{house_id}.xml")
        # ------------------------------------------------------------------------------------------

        # Have to apply the defaults after the settle, bc when trying to settle while using margin
        # it results in some objects being placed weirdly. This could be bc mjspec is not respecting
        # the defaults, so we better settle and then make sure the defaults are ok
        self._apply_defaults_to_house_mjcf(self.cfg.target_dir / f"{house_id}.xml")
        if self.cfg.flags.export_ceiling and self.cfg.scene_type != SceneType.ITHOR:
            self._apply_defaults_to_house_mjcf(self.cfg.target_dir / f"{house_id}_ceiling.xml")

        if self.cfg.copy_original:
            original_path = self.cfg.target_dir / f"{house_id}.xml"
            copy_path = self.cfg.target_dir / f"{house_id}_orig.xml"
            if original_path.is_file():
                shutil.copy(original_path, copy_path)

        scene_info_path = self.cfg.target_dir / f"{house_id}_metadata.json"
        with open(scene_info_path, "w") as fhandle:
            json.dump(asdict(self._scene_info), fhandle, indent=4)

        return self.spec

    def _process_settle(self, xml_house_path: Path) -> None:
        model = mj.MjModel.from_xml_path(xml_house_path.as_posix())
        data = mj.MjData(model)

        n_settle_steps = int(self.cfg.settle_time / 0.002)

        mj.mj_resetData(model, data)
        mj.mj_step(model, data, nstep=n_settle_steps)

        settle_poses: dict[str, tuple[np.ndarray, np.ndarray]] = dict()
        for body_id in range(model.nbody):
            name = model.body(body_id).name
            dofadr = model.body(body_id).dofadr.item()
            dofnum = model.body(body_id).dofnum.item()

            if dofadr == -1 or dofnum < 6:
                continue

            pos, quat = data.xpos[body_id].copy(), data.xquat[body_id].copy()
            settle_poses[name] = (pos, quat)

        tree = ET.parse(xml_house_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        assert worldbody is not None

        root_bodies = worldbody.findall("body")
        for root_body in root_bodies:
            name = root_body.attrib.get("name", "")
            if name in settle_poses:
                root_body.attrib["pos"] = " ".join(map(str, settle_poses[name][0]))
                root_body.attrib["quat"] = " ".join(map(str, settle_poses[name][1]))

        tree.write(xml_house_path)

        if self.cfg.flags.export_ceiling and self.cfg.scene_type != SceneType.ITHOR:
            xml_house_path_ceiling = xml_house_path.parent / f"{xml_house_path.stem}_ceiling.xml"
            tree = ET.parse(xml_house_path_ceiling)
            root = tree.getroot()
            worldbody = root.find("worldbody")
            assert worldbody is not None

            root_bodies = worldbody.findall("body")
            for root_body in root_bodies:
                name = root_body.attrib.get("name", "")
                if name in settle_poses:
                    root_body.attrib["pos"] = " ".join(map(str, settle_poses[name][0]))
                    root_body.attrib["quat"] = " ".join(map(str, settle_poses[name][1]))

            tree.write(xml_house_path_ceiling)

    def _apply_defaults_to_house_mjcf(self, xml_house_path: Path) -> None:
        if not xml_house_path.is_file():
            return

        tree = ET.parse(xml_house_path)
        root = tree.getroot()

        assert self.spec_opts is not None
        visual_def = self.spec_opts.find_default(VISUAL_CLASS)
        dynamic_def = self.spec_opts.find_default(DYNAMIC_CLASS)
        structural_def = self.spec_opts.find_default(STRUCTURAL_CLASS)
        structural_wall_def = self.spec_opts.find_default(STRUCTURAL_WALL_CLASS)
        articulable_dynamic_def = self.spec_opts.find_default(ARTICULABLE_DYNAMIC_CLASS)

        # Have to manually patch the defaults, as in the previous version of mujoco we used it
        # didn't respect the defaults and gave use some weird results. Also, could be bc the assets
        # might not follow the same defaults structure in some cases, so mjspec will create
        # duplicates for these cases
        def patch_defaults(element: ET.Element) -> None:
            classname = element.attrib.get("class", "")
            if classname == VISUAL_CLASS:
                geom = element.find("geom")
                if geom is not None:
                    geom.attrib["conaffinity"] = str(visual_def.geom.conaffinity)
                    geom.attrib["contype"] = str(visual_def.geom.contype)
                    geom.attrib["group"] = str(visual_def.geom.group)
                    geom.attrib["mass"] = str(visual_def.geom.mass)
            elif classname == DYNAMIC_CLASS:
                geom = element.find("geom")
                if geom is not None:
                    geom.attrib["conaffinity"] = str(dynamic_def.geom.conaffinity)
                    geom.attrib["contype"] = str(dynamic_def.geom.contype)
                    geom.attrib["group"] = str(dynamic_def.geom.group)
                    geom.attrib["friction"] = " ".join(map(str, dynamic_def.geom.friction.tolist()))
                    geom.attrib["solref"] = " ".join(map(str, dynamic_def.geom.solref.tolist()))
                    geom.attrib["solimp"] = " ".join(map(str, dynamic_def.geom.solimp.tolist()))
                    geom.attrib["density"] = str(dynamic_def.geom.density)
                    # geom.attrib["margin"] = str(DYNAMIC_OBJ_GEOMS_MARGIN)
            elif classname == STRUCTURAL_CLASS:
                geom = element.find("geom")
                if geom is not None:
                    geom.attrib["conaffinity"] = str(structural_def.geom.conaffinity)
                    geom.attrib["contype"] = str(structural_def.geom.contype)
                    geom.attrib["group"] = str(structural_def.geom.group)
                    geom.attrib["friction"] = " ".join(
                        map(str, structural_def.geom.friction.tolist())
                    )
                    geom.attrib["solref"] = " ".join(map(str, structural_def.geom.solref.tolist()))
                    geom.attrib["solimp"] = " ".join(map(str, structural_def.geom.solimp.tolist()))
                    geom.attrib["density"] = str(structural_def.geom.density)
                    # geom.attrib["margin"] = str(DYNAMIC_OBJ_GEOMS_MARGIN)
            elif classname == STRUCTURAL_WALL_CLASS:
                geom = element.find("geom")
                if geom is not None:
                    geom.attrib["group"] = str(structural_wall_def.geom.group)
            elif classname == ARTICULABLE_DYNAMIC_CLASS:
                geom = element.find("geom")
                if geom is not None:
                    geom.attrib["conaffinity"] = str(articulable_dynamic_def.geom.conaffinity)
                    geom.attrib["contype"] = str(articulable_dynamic_def.geom.contype)
                    geom.attrib["group"] = str(articulable_dynamic_def.geom.group)
                    geom.attrib["friction"] = " ".join(
                        map(str, articulable_dynamic_def.geom.friction.tolist())
                    )
                    geom.attrib["solref"] = " ".join(
                        map(str, articulable_dynamic_def.geom.solref.tolist())
                    )
                    geom.attrib["solimp"] = " ".join(
                        map(str, articulable_dynamic_def.geom.solimp.tolist())
                    )
                    geom.attrib["density"] = str(articulable_dynamic_def.geom.density)
                    # geom.attrib["margin"] = str(DYNAMIC_OBJ_GEOMS_MARGIN)

            for child in element.findall("default"):
                patch_defaults(child)

        base_default = root.find("default")
        if base_default is not None:
            patch_defaults(base_default)

        tree.write(xml_house_path)

    def _apply_patches_to_mjcf_house(self, xml_house_path: Path) -> None:
        tree = ET.parse(xml_house_path)
        root = tree.getroot()

        top_level_defaults = root.findall("default")
        for top_default in top_level_defaults:
            child_defaults = top_default.findall("default")
            to_remove = []
            for child_def in child_defaults:
                if "class" not in child_def.attrib:
                    to_remove.append(child_def)
            for def_to_remove in to_remove:
                top_default.remove(def_to_remove)

        asset_elm = root.find("asset")
        if asset_elm is not None:
            for texture_elm in asset_elm.findall("texture"):
                if texture_elm.attrib.get("type", "2d") == "skybox":
                    texture_elm.attrib["gridlayout"] = "LFRB.D.."
                    break

        # Set options, even though these might be the defaults and mjspec removes them
        compiler_elm = root.find("compiler")
        if compiler_elm is not None:
            compiler_elm.attrib["autolimits"] = "true"
            compiler_elm.attrib["boundmass"] = "0"
            compiler_elm.attrib["balanceinertia"] = "true"
        option_elm = root.find("option")
        if option_elm is not None:
            flag_elm = option_elm.find("flag")
            if flag_elm is not None:
                flag_elm.attrib["contact"] = "enable"
                flag_elm.attrib["warmstart"] = "enable"

        tree.write(xml_house_path)

    def load_from_json_path(
        self,
        thor_house_path: Path,
        target_dir: Path,
        house_id: str,
        stability_params: dict[str, float] = {},
    ) -> mj.MjSpec | None:
        if not thor_house_path.is_file():
            log.error(f"Path to house json '{thor_house_path.as_posix()}' is not valid")
            return

        with open(thor_house_path, "r") as fhandle:
            thor_house = json.load(fhandle)
        return self.load_from_json(
            thor_house=thor_house,
            target_dir=target_dir,
            house_id=house_id,
            stability_params=stability_params,
        )

    def _get_object_type(self, asset_id: str) -> SceneObjectType:
        if asset_id.lower() in self._cache_id_to_path:
            return SceneObjectType.THOR_OBJ
        elif self.cfg.objaverse_mjcf_dir and (self.cfg.objaverse_mjcf_dir / asset_id).is_dir():
            return SceneObjectType.OBJAVERSE_OBJ
        return SceneObjectType.CUSTOM_OBJ

    def _get_thor_model_path(
        self, asset_id: str, object_id: str, json_data: dict[str, Any] = {}
    ) -> tuple[Path | None, bool]:
        if asset_id.lower() not in self._cache_id_to_path:
            msg = f"Thor asset with assetId={asset_id}, object_id={object_id} not found in cache"
            self._logs["errors"].append(msg)
            raise FileNotFoundError(msg)

        model_info = self._cache_id_to_path[asset_id.lower()]
        if not model_info.has_any_model():
            msg = f"Cached Thor asset with id={asset_id} doesn't have any valid model"
            self._logs["errors"].append(msg)
            raise ValueError(msg)

        # Should change this to use only asset_id when we next re-export procthor-10k after
        # we make sure that doesn't make anything that was prim alredy back to mesh
        id_to_use = (
            object_id
            if self.cfg.scene_type not in {SceneType.HOLODECK, SceneType.PROCTHOR_OBJAVERSE}
            else asset_id
        )
        can_use_mesh = not must_use_prim(id_to_use)

        # This is a fix to make doors that are not openable primitive, not mesh, bc we are not going
        # to interact with those, so we can have fewer mesh geoms
        if "doorway_double" in asset_id.lower() and not json_data.get("openable", False):
            can_use_mesh = False

        if can_use_mesh and (model_info.mesh_path is not None) and model_info.mesh_path.is_file():
            return model_info.mesh_path, True
        elif (model_info.prim_path is not None) and model_info.prim_path.is_file():
            return model_info.prim_path, False
        return model_info.fallback_path, False

    def add_object_from_mjcf_model(  # noqa: PLR0911
        self,
        object_json: dict[str, Any],
        position: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
        parent_hash: str | None = None,
    ) -> SceneObjectInfo | None:
        assert self.spec is not None
        assert self.cfg.target_dir is not None
        assert self.cfg.target_assets_dir is not None

        asset_id = object_json.get("assetId", "")
        object_id = object_json.get("id", "")
        scene_obj_type = self._get_object_type(asset_id)
        if asset_id == "" and object_id == "":
            msg = "Can't add obj with both assetId and objectId being empty"
            self._logs["errors"].append(msg)
            log.error(msg)
            return

        # There are some RoboThor assets that don't have good primitive colliders, so we'll just use
        # a good one from regular Thor as a replacement in the scene
        if asset_id in THOR_ASSETS_TO_SWAP:
            asset_id = THOR_ASSETS_TO_SWAP[asset_id]

        object_id = object_id if object_id else f"{asset_id}_{self.cfg.counts[scene_obj_type]}"

        mjcf_filepath: Path | None = None
        obj_type: str = object_json.get(
            "objectType", self.asset_id_to_object_type.get(asset_id, "")
        )
        match scene_obj_type:
            case SceneObjectType.THOR_OBJ:
                mjcf_filepath, _ = self._get_thor_model_path(asset_id, object_id, object_json)
            case SceneObjectType.OBJAVERSE_OBJ:
                assert self.cfg.objaverse_mjcf_dir is not None, "Must provide mjcf objaverse folder"
                mjcf_filepath = self.cfg.objaverse_mjcf_dir / asset_id / f"{asset_id}.xml"
                if mjcf_filepath is None or not mjcf_filepath.is_file():
                    mjcf_filepath = None
                    msg = f"Obja. with id '{asset_id}' not found in mjcf generates assets folder"
                    self._logs["errors"].append(msg)
                    log.error(msg)
            case SceneObjectType.CUSTOM_OBJ:
                obj_file_candidates = list(self.cfg.target_assets_dir.rglob(f"**/{asset_id}.obj"))
                if len(obj_file_candidates) > 0:
                    obj_filepath = obj_file_candidates[0]
                    mjcf_filepath_candidate = (
                        obj_filepath.parent / obj_filepath.stem / f"{obj_filepath.stem}.xml"
                    )
                    if mjcf_filepath_candidate.is_file():
                        mjcf_filepath = mjcf_filepath_candidate
                    else:
                        msg = f"Couldn't load custom ithor object with asset_id='{asset_id}'"
                        self._logs["warnings"].append(msg)
                        log.warning(msg)
            case _:
                msg = f"Couldn't add object with assetId={asset_id}, object_id={object_id}"
                self._logs["errors"].append(msg)
                log.error(msg)
                return

        if scene_obj_type == SceneObjectType.THOR_OBJ and obj_type.lower() in THOR_ASSETS_TO_SKIP:
            return
        if scene_obj_type == SceneObjectType.THOR_OBJ and asset_id in THOR_ASSETS_INVALID_MUJOCO:
            return

        if scene_obj_type == SceneObjectType.THOR_OBJ and not self.cfg.flags.export_thor_objects:
            return
        if scene_obj_type == SceneObjectType.OBJAVERSE_OBJ and not self.cfg.flags.export_objaverse:
            return
        if scene_obj_type == SceneObjectType.CUSTOM_OBJ and not self.cfg.flags.export_custom_ithor:
            return

        if mjcf_filepath is None:
            return

        model_spec: mj.MjSpec = mj.MjSpec.from_file(mjcf_filepath.as_posix())

        relpath_to_target_dir = Path(
            os.path.relpath(mjcf_filepath.parent, start=self.cfg.target_dir)
        )

        # Cache all assets data, grab only the meshes that are needed ------------------------------
        self._collect_assets_textures_from_mjcf_object(model_spec, relpath_to_target_dir)
        self._collect_assets_materials_from_mjcf_object(model_spec, scene_obj_type)
        self._collect_assets_meshes_from_mjcf_object(model_spec, relpath_to_target_dir)
        # ------------------------------------------------------------------------------------------

        # Naming setup according to our naming convention ------------------------------------------
        parts = object_id.replace("|surface", "").split("|")
        room_id_str = parts[1] if len(parts) > 1 else "0"  # noqa: PLR2004
        room_id = int(room_id_str) if room_id_str.isdigit() else 0
        hash_of_type = (
            generate_object_hash(asset_id)
            if scene_obj_type != SceneObjectType.OBJAVERSE_OBJ
            else asset_id
        )

        self._counts[hash_of_type] += 1

        lemma = ""
        if obj_type in THOR_TYPE_TO_LEMMA:
            lemma = THOR_TYPE_TO_LEMMA[obj_type].replace("_", "")
        else:
            if (
                self.cfg.scene_type == SceneType.HOLODECK
                and scene_obj_type == SceneObjectType.OBJAVERSE_OBJ
            ):
                lemma = "obja"
            else:
                lemma = obj_type.lower().split("_")[0].replace("_", "")
        root_name = generate_body_name(lemma, hash_of_type, self._counts[hash_of_type], 0, room_id)

        body_counter = 0
        old_to_new_names_map: dict[str, str] = {}

        def collect_body_names_recursively(body_spec: mj.MjsBody) -> None:
            nonlocal hash_of_type, body_counter, room_id
            old_to_new_names_map[body_spec.name] = generate_body_name(
                lemma, hash_of_type, self._counts[hash_of_type], body_counter, room_id
            )
            body_counter += 1
            for child_spec in body_spec.bodies:
                collect_body_names_recursively(child_spec)

        collect_body_names_recursively(model_spec.worldbody.first_body())
        # ------------------------------------------------------------------------------------------

        # Handle joints according to some predefined rules -----------------------------------------

        # Remove free joint if it's frozen in space
        frozen_in_space: bool = is_frozen_in_space(obj_type, asset_id)
        kinematic: bool = object_json.get("kinematic", False)
        if frozen_in_space or kinematic:
            for joint in model_spec.joints:
                assert isinstance(joint, mj.MjsJoint)
                if joint.type == mj.mjtJoint.mjJNT_FREE:
                    model_spec.delete(joint)
        else:
            has_free_joint = any(
                [True for joint in model_spec.joints if joint.type == mj.mjtJoint.mjJNT_FREE]
            )
            if not has_free_joint:
                model_spec.worldbody.first_body().add_freejoint(name="freejnt")

        for joint in model_spec.joints:
            assert isinstance(joint, mj.MjsJoint)
            if joint.type == mj.mjtJoint.mjJNT_FREE:
                joint.stiffness = 0.0
                joint.damping = self.cfg.param_freejoint_damping
                joint.frictionloss = self.cfg.param_freejoint_frictionloss
                joint.armature = 0.0
                break

        # Remove all joints in case of a very specific category
        if obj_type in TYPES_TO_REMOVE_ALL_JOINTS:
            for joint in model_spec.joints:
                assert isinstance(joint, mj.MjsJoint)
                if joint.type == mj.mjtJoint.mjJNT_FREE:
                    continue
                model_spec.delete(joint)

        # For the case of the light switch category, remove all hinge joints for now
        if obj_type.lower().startswith("lightswitch"):
            for joint in model_spec.joints:
                assert isinstance(joint, mj.MjsJoint)
                if joint.type == mj.mjtJoint.mjJNT_HINGE:
                    model_spec.delete(joint)

        openness: float = object_json.get("openness", -1.0)
        if kinematic:

            def make_kinematic(body_spec: mj.MjsBody) -> None:
                for joint in body_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    model_spec.delete(joint)

            root_body_spec = model_spec.worldbody.first_body()
            make_kinematic(root_body_spec)
            for child_spec in root_body_spec.bodies:
                make_kinematic(child_spec)
        elif openness >= 0:
            ### unlike doors, some objects's zero position is when it's fully open
            # (i.e. laptop_full, box_full,)
            # so we need to invert the openness if this assumption is not met (i.e. book)
            openness = 1 - openness if obj_type.lower() not in {"laptop", "box"} else openness
            root_body_spec = model_spec.worldbody.first_body()
            for child_body_spec in root_body_spec.bodies:
                assert isinstance(child_body_spec, mj.MjsBody)
                if len(child_body_spec.joints) == 0:
                    continue
                if len(child_body_spec.joints) > 1:
                    raise ValueError(
                        f"Model with asset_id={asset_id} has a body with multiple joints"
                    )
                joint_spec: mj.MjsJoint = child_body_spec.joints[0]
                if joint_spec.type == mj.mjtJoint.mjJNT_FREE:
                    continue
                child_quat = child_body_spec.quat.copy()
                jnt_axis = joint_spec.axis.copy()
                jnt_range = joint_spec.range.copy()

                nonzero_index = np.nonzero(jnt_range)[0]
                assert len(nonzero_index) == 1, (
                    f"Joint '{joint_spec.name}': only one axis should be non-zero"
                )

                nonzero_index = nonzero_index.item()
                adjusted_openness = 1.0 - openness if nonzero_index == 1 else openness

                joint_pos = adjusted_openness * (jnt_range[1] - jnt_range[0]) + jnt_range[0]
                if joint_spec.type == mj.mjtJoint.mjJNT_HINGE:
                    before_rot = R.from_quat(child_quat, scalar_first=True)
                    after_rot = before_rot * R.from_rotvec(joint_pos * jnt_axis)
                    child_body_spec.quat = after_rot.as_quat(scalar_first=True)
                    joint_spec.ref = joint_pos
                elif joint_spec.type == mj.mjtJoint.mjJNT_SLIDE:
                    before_pos = child_body_spec.pos.copy()
                    after_pos = before_pos + jnt_axis * joint_pos
                    child_body_spec.pos = after_pos

        # ------------------------------------------------------------------------------------------

        # Grab the transform for this object -------------------------------------------------------
        pos: np.ndarray | None = None
        if "position" in object_json:
            pos = np.array(unity_to_mj_pos([object_json["position"][k] for k in "xyz"]))
        elif position is not None:
            pos = position
        assert pos is not None, f"Got 'none' when getting the position of object '{object_id}'"

        quat: np.ndarray | None = None
        if "rotation" in object_json:
            unity_rotmat = R.from_euler(
                "zxy", [object_json["rotation"][k] for k in "xyz"], degrees=True
            ).as_matrix()
            if "stoveknob" in asset_id.lower():
                rot_json = object_json["rotation"]
                unity_rotmat = R.from_euler(
                    "zxy", [rot_json["x"], -rot_json["y"], -rot_json["z"]], degrees=True
                ).as_matrix()
            quat = R.from_matrix(T_FIX_ROTMAT @ unity_rotmat).as_quat(scalar_first=False)  # T_T
        elif quaternion is not None:
            quat = quaternion

        if scene_obj_type == SceneObjectType.OBJAVERSE_OBJ and self.cfg.objaverse_data_dir:
            msgpack_filepath = self.cfg.objaverse_data_dir / asset_id / f"{asset_id}.msgpack.gz"
            if msgpack_filepath.is_file():
                with gzip.open(msgpack_filepath, "rb") as fhandle:
                    msgpack_data = msgpack.unpackb(fhandle.read(), raw=False)
                # Have to apply the opposite rotation from the original asset, as the original mjcf
                # was created using this rotation, but this was already applied to the vertex data
                # of the objaverse asset, so we have to cancel it
                if "yRotOffset" in msgpack_data:
                    y_rot_offset = msgpack_data["yRotOffset"]
                    extra_rot = R.from_rotvec([0.0, 0.0, -y_rot_offset], degrees=True)
                    r_quat = R.from_quat(quat, scalar_first=True)
                    r_quat_new = extra_rot * r_quat
                    quat = r_quat_new.as_quat(scalar_first=True)

        assert quat is not None, f"Got 'none' when getting the quaternion of object '{object_id}'"

        # This is a fix for assets of type 'Desk_306_1' and 'Toilet_2', bc it seems that these have
        # an extra offset when used in procthor-like scenes
        FIX_OFFSETS = {
            "Toilet_2": np.array([0.0, 0.0, 0.15]),
            "Desk_306_1": np.array([-0.2, 0.0, 0.0]),
        }
        if self.cfg.scene_type in {
            SceneType.PROCTHOR_10K,
            SceneType.PROCTHOR_OBJAVERSE,
            SceneType.HOLODECK,
        }:
            if asset_id in FIX_OFFSETS:
                pos_offset = FIX_OFFSETS[asset_id].copy()
                root_body = model_spec.worldbody.first_body()
                for child_body in root_body.bodies:
                    assert isinstance(child_body, mj.MjsBody)
                    child_body.pos += pos_offset

        FIX_OFFSETS_HOLODECK = {
            "Coffee_Table_222_1": np.array([0.0, 0.0, 0.02]),
            "Dresser_413_1": np.array([0.0, 0.0, 0.008]),
            "Dresser_301_1": np.array([0.0, 0.0, 0.004]),
            "Side_Table_322_1": np.array([0.0, 0.0, 0.0265]),
            "Side_Table_311_2_2": np.array([0.0, 0.0, 0.0058]),
        }
        if self.cfg.scene_type == SceneType.HOLODECK:
            if asset_id in FIX_OFFSETS_HOLODECK:
                pos_offset = FIX_OFFSETS_HOLODECK[asset_id].copy()
                root_body = model_spec.worldbody.first_body()
                for child_body in root_body.bodies:
                    assert isinstance(child_body, mj.MjsBody)
                    child_body.pos += pos_offset
        # ------------------------------------------------------------------------------------------

        # Setup the names using our naming convention ----------------------------------------------
        tag_counter = Counter()
        obj_name_map = ObjectNameMapping()
        change_name_recursively(
            model_spec.worldbody.first_body(),
            hash_of_type=hash_of_type,
            count_of_type=self._counts[hash_of_type],
            lemma=lemma,
            counter=tag_counter,
            asset_id=asset_id,
            object_id=object_id,
            room_id=room_id,
            name_map=obj_name_map,
        )
        # ------------------------------------------------------------------------------------------

        # Assing classes to geoms according to our predefined rules --------------------------------
        self._setup_geoms_classes(
            model_spec.worldbody.first_body(),
            frozen_in_space,
            kinematic,
            asset_id,
            True,
        )
        # ------------------------------------------------------------------------------------------

        # Grab all contact exclusions and renamed them when adding it back -------------------------
        for exclusion in model_spec.excludes:
            assert isinstance(exclusion, mj.MjsExclude)
            if (
                exclusion.bodyname1 not in old_to_new_names_map
                or exclusion.bodyname2 not in old_to_new_names_map
            ):
                msg = f"Object {object_id} of asset type {asset_id} has issues with exclusion pairs"
                self._logs["warnings"].append(msg)
                log.warning(msg)
                continue
            new_bodyname1 = old_to_new_names_map[exclusion.bodyname1]
            new_bodyname2 = old_to_new_names_map[exclusion.bodyname2]
            key = f"{new_bodyname1}-{new_bodyname2}"
            key_reverse = f"{new_bodyname2}-{new_bodyname1}"
            if key in self._cache_exclusion_pairs or key_reverse in self._cache_exclusion_pairs:
                continue
            self._cache_exclusion_pairs.add(key)
            self._cache_exclusion_pairs.add(key_reverse)
            self.spec.add_exclude(
                name=f"mjt_pair_{len(self.spec.excludes)}",
                bodyname1=new_bodyname1,
                bodyname2=new_bodyname2,
            )
        for exclusion in model_spec.excludes:
            assert isinstance(exclusion, mj.MjsExclude)
            model_spec.delete(exclusion)
        # ------------------------------------------------------------------------------------------

        if scene_obj_type == SceneObjectType.CUSTOM_OBJ:
            self._get_custom_thor_defaults(model_spec, asset_id)

        # Use XML|ElementTree to remove empty defaults, it seems that mjspec doesn't
        # allow this. The thing is that there are some defaults that have different settings, so
        # those are not actually the same, and that's likely the reason why it keeps the copies
        def_handle = model_spec.find_default("/")
        if def_handle is not None:
            model_spec.delete(def_handle)

        # We have to manually add the materials that are going to be used, otherwise mjspec will do
        # the wrong thing, and copy all other stuff that we might not need. Could have been fixed
        # in the latest versions, or could use the 'assets' argument when attaching a spec

        root_body = model_spec.worldbody.first_body()
        frame_handle = self.spec.worldbody.add_frame()
        root_body.pos = pos.copy()
        root_body.quat = quat.copy()
        frame_handle.attach_body(root_body)

        match scene_obj_type:
            case SceneObjectType.THOR_OBJ:
                self.cfg.counts[SceneObjectType.THOR_OBJ] += 1
            case SceneObjectType.OBJAVERSE_OBJ:
                self.cfg.counts[SceneObjectType.OBJAVERSE_OBJ] += 1
            case SceneObjectType.CUSTOM_OBJ:
                self.cfg.counts[SceneObjectType.CUSTOM_OBJ] += 1

        object_info = SceneObjectInfo(
            hash_name=root_name,
            asset_id=asset_id,
            object_id=object_id,
            category=obj_type,
            name_map=obj_name_map,
            object_enum=scene_obj_type,
            is_static=frozen_in_space or kinematic or obj_type in TYPES_TO_REMOVE_ALL_JOINTS,
            parent=parent_hash if parent_hash is not None else "",
            room_id=room_id,
        )
        self._scene_info.objects[root_name] = object_info

        if "children" in object_json:
            for child_json in object_json["children"]:
                if isinstance(child_json, dict):
                    child_info = self.add_object_from_mjcf_model(child_json, None, None, root_name)
                    if child_info:
                        object_info.children.append(child_info.hash_name)

        return object_info

    def _get_custom_thor_defaults(self, model_spec: mj.MjSpec, asset_id: str) -> None:
        category = next(
            (
                cat
                for cat in ITHOR_CUSTOM_OBJ_CATEGORIES
                if asset_id.lower().startswith(cat.lower())
            ),
            None,
        )
        if category is None:
            return

        match category.lower():
            case "cabinet":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 0.0
                    joint.damping = 0.1
                    joint.armature = 0.01
                    joint.frictionloss = 1
            case "drawer":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 0.0
                    joint.damping = 0.025
                    joint.armature = 0.05
                    joint.frictionloss = 1.0
            case "showerdoor":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 0.0
                    joint.damping = 0.1
                    joint.armature = 0.05
                    joint.frictionloss = 0.5
            case "oven":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 1.0
                    joint.damping = 10.0
                    joint.armature = 0.1
                    joint.frictionloss = 10.0
            case "dishwasher":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 1.0
                    joint.damping = 10.0
                    joint.armature = 0.1
                    joint.frictionloss = 10.0
            case "stoveknob":
                for joint in model_spec.joints:
                    assert isinstance(joint, mj.MjsJoint)
                    if joint.type not in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                        continue
                    joint.stiffness = 0.0
                    joint.damping = 0.1
                    joint.armature = 0.1
                    joint.frictionloss = 0.5

    def _collect_assets_textures_from_mjcf_object(
        self, model_spec: mj.MjSpec, relpath_to_target_dir: Path
    ) -> None:
        assert self.spec is not None
        textures_to_delete: list[mj.MjsTexture] = []
        for texture in model_spec.textures:
            assert isinstance(texture, mj.MjsTexture)

            if texture.name in self._assets_cache.textures:
                texture_info = self._assets_cache.textures[texture.name]
            else:
                texture_info = Texture2dInfo(
                    name=texture.name, file=relpath_to_target_dir / texture.file
                )
                self._assets_cache.textures[texture.name] = texture_info

            if (
                self.spec.texture(texture.name) is None
                and texture.name not in self._textures_added_set
            ):
                self.spec.add_texture(
                    name=texture_info.name,
                    file=texture_info.file.as_posix(),
                    type=mj.mjtTexture.mjTEXTURE_2D,
                )
                self._textures_added_set.add(texture.name)
            textures_to_delete.append(texture)

        # Delete so we don't copy it when attaching
        for texture in textures_to_delete:
            model_spec.delete(texture)

    def _collect_assets_materials_from_mjcf_object(
        self, model_spec: mj.MjSpec, scene_obj_type: SceneObjectType
    ) -> None:
        assert self.spec is not None
        materials_to_delete: list[mj.MjsMaterial] = []
        for material in model_spec.materials:
            assert isinstance(material, mj.MjsMaterial)
            mat_name = material.name
            if scene_obj_type in {SceneObjectType.THOR_OBJ, SceneObjectType.CUSTOM_OBJ}:
                if mat_name.startswith("material_"):
                    mat_name = mat_name[len("material_") :]

            if mat_name in self._assets_cache.materials:
                material_info = copy.deepcopy(self._assets_cache.materials[mat_name])
            else:
                material_info = MaterialInfo(name=material.name)
                material_info.shininess = material.shininess
                material_info.specular = material.specular
                material_info.rgba = material.rgba.copy()
                self._assets_cache.materials[mat_name] = material_info

            tex_id = material.textures[mj.mjtTextureRole.mjTEXROLE_RGB.value]
            if tex_id != "" and tex_id in self._assets_cache.textures:
                material_info.texture = self._assets_cache.textures[tex_id]

            if (
                self.spec.material(material.name) is None
                and material.name not in self._materials_added_set
            ):
                mat_spec = self.spec.add_material(name=material.name)
                self._materials_added_set.add(material.name)
                if material_info.texture is not None:
                    texture_info = material_info.texture
                    mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB.value] = texture_info.name
                    if (
                        self.spec.texture(texture_info.name) is None
                        and texture_info.name not in self._textures_added_set
                    ):
                        self.spec.add_texture(
                            name=texture_info.name,
                            file=texture_info.file.as_posix(),
                            type=mj.mjtTexture.mjTEXTURE_2D,
                        )
                if material_info.rgba is not None:
                    mat_spec.rgba = (
                        material.rgba
                        if material_info.rgba[3] < THRESHOLD_TRANSPARENCY
                        else material_info.rgba
                    )
                else:
                    mat_spec.rgba = material.rgba
            materials_to_delete.append(material)

        # Delete so we don't copy it when attaching
        for material in materials_to_delete:
            model_spec.delete(material)

    def _collect_assets_meshes_from_mjcf_object(
        self, model_spec: mj.MjSpec, relpath_to_target_dir: Path
    ) -> None:
        assert self.spec is not None
        for mesh in model_spec.meshes:
            assert isinstance(mesh, mj.MjsMesh)
            if mesh.name not in self._assets_cache.meshes:
                mesh_info = MeshInfo(
                    name=mesh.name,
                    file=relpath_to_target_dir / mesh.file,
                    scale=mesh.scale.copy(),
                    inertia=mesh.inertia,
                )
                self._assets_cache.meshes[mesh_info.name] = mesh_info
                if self.spec.mesh(mesh_info.name) is not None:
                    msg = f"Mesh '{mesh_info.name}' already added to the spec"
                    self._logs["warnings"].append(msg)
                    log.warning(msg)
                else:
                    self.spec.add_mesh(
                        name=mesh_info.name,
                        file=mesh_info.file.as_posix(),
                        scale=mesh_info.scale,
                        inertia=mesh_info.inertia,
                    )
            model_spec.delete(mesh)

    def _setup_geoms_classes(
        self,
        body_spec: mj.MjsBody,
        frozen_in_space: bool,
        kinematic: bool,
        asset_id: str,
        use_margin: bool,
    ) -> None:
        assert self.spec is not None
        main_def = self.spec.find_default("main")
        for geom in body_spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.classname.name in VALID_VISUAL_CLASSES:
                set_defaults(geom, self.spec.find_default(VISUAL_CLASS), main_def)
            elif (
                geom.classname.name in VALID_DYNAMIC_CLASSES or "collision" in geom.meshname.lower()
            ):
                if frozen_in_space or kinematic:
                    in_articulated_category = any(
                        [cat in asset_id.lower() for cat in ARTICULABLE_DYNAMIC_CATEGORIES]
                    )
                    if in_articulated_category:
                        set_defaults(
                            geom, self.spec.find_default(ARTICULABLE_DYNAMIC_CLASS), main_def
                        )
                    else:
                        set_defaults(geom, self.spec.find_default(STRUCTURAL_CLASS), main_def)
                else:
                    set_defaults(geom, self.spec.find_default(DYNAMIC_CLASS), main_def)
                if use_margin and geom.type == mj.mjtGeom.mjGEOM_MESH:
                    geom.margin = self.cfg.param_geom_margin
            else:
                msg = f"Geom '{geom.name}' of class '{geom.classname.name}' not taken into account"
                self._logs["warnings"].append(msg)
                log.warning(msg)

        for child_spec in body_spec.bodies:
            assert isinstance(child_spec, mj.MjsBody)
            self._setup_geoms_classes(child_spec, frozen_in_space, kinematic, asset_id, use_margin)

    def add_light(self, light_json: dict[str, Any]) -> None:
        assert self.spec is not None

        light_type = light_json.get("type", "point").lower()
        match light_type:
            case "point":
                position = np.array(unity_to_mj_pos([light_json["position"][k] for k in "xyz"]))
                diffuse = np.array([light_json["rgb"][k] for k in "rgb"])
                l_range = light_json["range"]

                self.spec.worldbody.add_light(
                    pos=position,
                    diffuse=diffuse,
                    range=l_range,
                    type=mj.mjtLightType.mjLIGHT_POINT,
                )
            case "spot":
                position = np.array(unity_to_mj_pos([light_json["position"][k] for k in "xyz"]))
                direction = np.array(unity_to_mj_pos([light_json["direction"][k] for k in "xyz"]))
                direction /= np.linalg.norm(direction)
                diffuse = np.array([light_json["rgb"][k] for k in "rgb"])
                l_range = light_json["range"]

                self.spec.worldbody.add_light(
                    pos=position,
                    diffuse=diffuse,
                    dir=direction,
                    range=l_range,
                    type=mj.mjtLightType.mjLIGHT_SPOT,
                )

    def add_wall(self, wall_json: dict[str, Any]) -> None:
        assert self.spec is not None
        assert self.cfg.target_dir is not None
        assert self.cfg.target_assets_dir is not None
        main_def = self.spec.find_default("main")

        vertices = np.array([[vert[k] for k in "xyz"] for vert in wall_json["polygon"]])

        wall_name = wall_json.get("id", "unnamed")
        wall_room_id: int = 0
        if self.cfg.scene_type == SceneType.HOLODECK:
            pattern = r"^(\D+)(\d+)$"
            re_match = re.match(pattern, wall_json.get("roomId", ""))
            if re_match:
                wall_room_id = int(re_match.group(2))
                wall_name = f"wall_{wall_room_id}_{self.cfg.counts[SceneObjectType.WALL]}"
        else:
            wall_room_str_parts: str = wall_json.get("roomId", "").split("|")
            wall_room_id_str: str = wall_room_str_parts[-1] if len(wall_room_str_parts) > 0 else "0"
            wall_room_id = int(wall_room_id_str) if wall_room_id_str.isdigit() else 0
            wall_name = f"wall_{wall_room_id}_{self.cfg.counts[SceneObjectType.WALL]}"

        if "id" not in wall_json:
            raise ValueError("Got a wall that is not associated with an id. Wait what????")

        self._name_mapping.walls[wall_json["id"]] = wall_name

        mesh_wall: o3d.geometry.TriangleMesh | None = None
        mesh_filepath = self.cfg.target_assets_dir / f"{wall_name}.obj"
        if not mesh_filepath.is_file() or self.cfg.overwrite_mesh:
            p0 = vertices[0]
            p1 = vertices[1]
            dir = np.sign(np.sum(p0 - p1))

            # Holodeck wall polygons follow a diferent order, so make them the same as the ones in
            # procthor houses to keep the same code for generating the walls
            if self.cfg.scene_type == SceneType.HOLODECK:
                vertices = vertices[[0, 3, 1, 2]]

            if vertices[0][1] > vertices[2][1] and vertices[1][1] > vertices[3][1]:
                vertices = np.array([vertices[2], vertices[3], vertices[0], vertices[1]])

            if vertices[0][2] > vertices[1][2] and vertices[2][2] > vertices[3][2]:
                vertices = np.array([vertices[1], vertices[0], vertices[3], vertices[2]])

            if vertices[0][0] > vertices[1][0] and vertices[2][0] > vertices[3][0]:
                vertices = np.array([vertices[1], vertices[0], vertices[3], vertices[2]])

            mesh_wall = create_wall_mesh(vertices, DEFAULT_MIN_WALL_THICKNESS * 5, dir=dir)
            with stdout_redirected():
                o3d.io.write_triangle_mesh(mesh_filepath, mesh_wall)
        else:
            mesh_wall = o3d.io.read_triangle_mesh(mesh_filepath)

        assert mesh_wall is not None, (
            f"Must have a valid 'mesh_wall' by now. Filepath={mesh_filepath.as_posix()}"
        )

        wall_mat_info: MaterialInfo | None = self._setup_material(wall_json, SceneObjectType.WALL)

        if self.spec.body(wall_name) is not None:
            raise RuntimeError(f"Wall '{wall_name}' already exists in the house spec")

        wall_body = self.spec.worldbody.add_body(name=wall_name)

        wall_mesh_name = wall_name
        if self.spec.mesh(wall_mesh_name) is None:
            mesh_rel_filepath = mesh_filepath.relative_to(self.cfg.target_dir)
            self.spec.add_mesh(
                name=wall_mesh_name, file=mesh_rel_filepath.as_posix(), scale=[1, 1, -1]
            )
        else:
            msg = f"The mesh associated with wall '{wall_mesh_name}' has alredy been created"
            self._logs["warnings"].append(msg)
            log.warning(msg)

        wall_visual_name = f"{wall_name}_visual_0"
        if self.spec.geom(wall_visual_name) is None:
            wall_geom = wall_body.add_geom(
                name=wall_visual_name,
                type=mj.mjtGeom.mjGEOM_MESH,
                meshname=wall_mesh_name,
                quat=R.from_matrix(T_TRANSFORM[:3, :3]).as_quat(scalar_first=True),
            )
            set_defaults(wall_geom, self.spec.find_default(VISUAL_CLASS), main_def)
            if wall_mat_info:
                wall_geom.material = wall_mat_info.name

        wall_collider_name = f"{wall_name}_collision_0"
        if self.spec.geom(wall_collider_name) is None:
            wall_structural_geom = wall_body.add_geom(
                name=wall_collider_name,
                type=mj.mjtGeom.mjGEOM_MESH,
                meshname=wall_mesh_name,
                quat=R.from_matrix(T_TRANSFORM[:3, :3]).as_quat(scalar_first=True),
            )
            set_defaults(
                wall_structural_geom, self.spec.find_default(STRUCTURAL_WALL_CLASS), main_def
            )
        else:
            msg = f"Tried adding wall called '{wall_name}', but it is already in the scene spec"
            self._logs["errors"].append(msg)
            log.error(msg)

        self.cfg.counts[SceneObjectType.WALL] += 1

        self._wall_cache[wall_name] = WallCachedInfo(
            filepath=mesh_filepath,
            vertices=vertices,
            mesh=mesh_wall,
            geom_name=wall_collider_name,
            visual_name=wall_visual_name,
            body_name=wall_name,
            room_id=wall_room_id,
            mat_info=wall_mat_info,
        )
        log.debug(f"Created wall '{wall_name}' @ {mesh_filepath.as_posix()}")

    def add_walls_uv(self, wall_ids: Sequence[str]) -> None:
        for wall_id in wall_ids:
            wall_name = self._name_mapping.walls.get(wall_id, "")
            if wall_name not in self._wall_cache:
                continue
            mesh_filename = self._wall_cache[wall_name].filepath
            self.create_uv_coordinates(mesh_filename, 1.0)

    def add_room(
        self,
        room_json: dict[str, Any],
        room_z: float = 0.0,
        name_prefix: str | None = None,
        material_name: str | None = None,
    ) -> None:
        assert self.spec is not None
        assert self.cfg.target_dir is not None
        assert self.cfg.target_assets_dir is not None
        main_def = self.spec.find_default("main")

        if "id" not in room_json:
            msg = "Tried to process a room that didn't have an 'id'"
            self._logs["errors"].append(msg)
            log.error(msg)
            return

        room_name = room_json["id"]
        if self.cfg.scene_type == SceneType.HOLODECK:
            pattern = r"^(\D+)(\d+)$"
            re_match = re.match(pattern, room_json["id"])
            if re_match:
                room_id: int = int(re_match.group(2))
                room_name = f"room_{room_id}" if name_prefix is None else f"{name_prefix}_{room_id}"
        else:
            room_str_parts: str = room_json.get("id", "").split("|")
            room_str_id: str = room_str_parts[-1] if len(room_str_parts) > 0 else "0"
            room_id: int = int(room_str_id) if room_str_id.isdigit() else 0
            room_name = f"room_{room_id}" if name_prefix is None else f"{name_prefix}_{room_id}"

        self._name_mapping.rooms[room_json["id"]] = room_name

        room_vertices = np.array(
            [[xyz[k] for k in "xyz"] for xyz in room_json.get("floorPolygon", [])]
        )

        if self.spec.body(room_name) is not None:
            raise RuntimeError(f"Room with name {room_name} already exists in the house spec")

        room_body = self.spec.worldbody.add_body(name=room_name)

        mesh_room: o3d.geometry.TriangleMesh | None = None
        mesh_filepath = self.cfg.target_assets_dir / f"{room_name}.obj"
        if not mesh_filepath.is_file() or self.cfg.overwrite_mesh:
            mesh_room = create_room_mesh(room_vertices, DEFAULT_MIN_WALL_THICKNESS * 5)
            with stdout_redirected():
                o3d.io.write_triangle_mesh(mesh_filepath.as_posix(), mesh_room)
            if self.cfg.room_opts.generate_uvs:
                self.create_uv_coordinates(mesh_filepath, self.cfg.room_opts.uv_scale_factor)
        else:
            mesh_room = o3d.io.read_triangle_mesh(mesh_filepath.as_posix())

        assert mesh_room is not None, (
            f"Must have a valid 'mesh_room' by now. Filepath={mesh_filepath.as_posix()}"
        )

        if material_name is not None:
            if "floorMaterial" not in room_json:
                room_json["floorMaterial"] = {}
            room_json["floorMaterial"]["name"] = material_name

        room_mat_info = self._setup_material(room_json, SceneObjectType.ROOM)

        room_mesh_name = room_name
        if self.spec.mesh(room_mesh_name) is None:
            mesh_rel_filepath = mesh_filepath.relative_to(self.cfg.target_dir)
            self.spec.add_mesh(
                name=room_mesh_name, file=mesh_rel_filepath.as_posix(), scale=[1, 1, -1]
            )
        else:
            msg = f"The mesh associated with room '{room_mesh_name}' has alredy been created"
            self._logs["warnings"].append(msg)
            log.warning(msg)

        room_geom_name = f"{room_name}_visual_0"
        if self.spec.geom(room_geom_name) is not None:
            raise RuntimeError(
                f"Geom {room_geom_name} for room {room_name} already exists in the house spec"
            )

        room_geom = room_body.add_geom(
            name=room_geom_name,
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=room_mesh_name,
            pos=[0, 0, room_z],
            quat=R.from_matrix(T_TRANSFORM[:3, :3]).as_quat(scalar_first=True),
        )
        set_defaults(room_geom, self.spec.find_default(VISUAL_CLASS), main_def)
        if room_mat_info:
            room_geom.material = room_mat_info.name

        self.cfg.counts[SceneObjectType.ROOM] += 1

        self._room_cache[room_name] = RoomCachedInfo(
            filepath=mesh_filepath,
            mesh=mesh_room,
            vertices=room_vertices,
        )
        log.debug(f"Created room '{room_name}' @ {mesh_filepath.as_posix()}")

    def add_door(self, door_json: dict[str, Any]) -> None:
        assert self.spec is not None
        assert self.cfg.target_dir is not None
        assert self.cfg.target_assets_dir is not None
        main_def = self.spec.find_default("main")

        door_id = door_json.get("id", "")
        if door_id == "":
            msg = "Tried to process a room that didn't have an 'id'"
            self._logs["errors"].append(msg)
            log.error(msg)
            return

        wall0_name, wall0_info = self._get_cached_wall_info(door_json["wall0"])
        wall1_name, _ = self._get_cached_wall_info(door_json["wall1"])

        use_same_room = False
        if self.cfg.scene_type == SceneType.HOLODECK:
            # There are some rooms that have "exterior" as value for "room0", so use this default
            # for the edge case (we have tested it and this default makes it work fine)
            if door_json["room0"] not in self._name_mapping.rooms:
                use_same_room = True

        _, room0_info = self._get_cached_room_info(
            door_json["room0" if not use_same_room else "room1"]
        )
        _, room1_info = self._get_cached_room_info(door_json["room1"])

        flip_2d_coord = should_flip_asset_direction(
            room0_info.vertices, room1_info.vertices, wall0_info.vertices
        )
        if use_same_room:
            flip_2d_coord = True

        hole_topleft_bottomright = np.array(
            [[xyz[k] for k in "xyz"] for xyz in door_json.get("holePolygon", [])]
        )
        pos: np.ndarray | None = None
        for wall_name, direction in zip([wall0_name, wall1_name], [1, -1]):
            geom_name = self._wall_cache[wall_name].geom_name
            geom_handle = self.spec.geom(geom_name)
            if geom_handle is None:
                raise ValueError(f"Couldn't find geom called '{geom_name}' in the current spec")
            self.spec.delete(geom_handle)

            visual_name = self._wall_cache[wall_name].visual_name
            visual_handle = self.spec.geom(visual_name)
            if visual_handle is None:
                raise ValueError(f"Couldn't find visual called '{visual_name}' in the current spec")
            self.spec.delete(visual_handle)

            body_name = self._wall_cache[wall_name].body_name
            body_handle = self.spec.body(body_name)
            if body_handle is None:
                raise ValueError(f"Couldn't find body called '{body_name}' in the current spec")

            mesh_handle = self.spec.mesh(wall_name)
            if mesh_handle is None:
                raise ValueError(f"Couldn't find mesh called '{wall_name}' in the current spec")
            self.spec.delete(mesh_handle)

            # Create a new wall witih hole
            wall_mesh_filepath = self._wall_cache[wall_name].filepath
            wall_mesh = self._wall_cache[wall_name].mesh

            updated_wall_mesh, asset_pos, asset_rot = make_hole_in_wall(
                wall_mesh=wall_mesh,
                hole_topleft=hole_topleft_bottomright[0],
                hole_bottomright=hole_topleft_bottomright[1],
                direction=direction,
                asset_position_2d=[
                    door_json["assetPosition"]["x"],
                    door_json["assetPosition"]["y"],
                ],
                type="door",
                flip_2d_coord=flip_2d_coord,
                fix_problematic_holodeck=(
                    self.cfg.scene_type in {SceneType.PROCTHOR_OBJAVERSE, SceneType.HOLODECK}
                ),
            )
            with stdout_redirected():
                o3d.io.write_triangle_mesh(wall_mesh_filepath, updated_wall_mesh)

            wall_mesh_name = wall_name
            if self.spec.mesh(wall_mesh_name) is None:
                mesh_rel_filepath = wall_mesh_filepath.relative_to(self.cfg.target_dir)
                self.spec.add_mesh(
                    name=wall_mesh_name, file=mesh_rel_filepath.as_posix(), scale=[1, 1, -1]
                )
            else:
                msg = f"The mesh associated with wall '{wall_mesh_name}' has alredy been created"
                self._logs["warnings"].append(msg)
                log.warning(msg)

            if self.spec.geom(visual_name) is None:
                wall_visual_geom = body_handle.add_geom(
                    name=visual_name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=wall_mesh_name,
                    quat=R.from_matrix(T_TRANSFORM[:3, :3]).as_quat(scalar_first=True),
                )
                set_defaults(wall_visual_geom, self.spec.find_default(VISUAL_CLASS), main_def)
                wall_mat_info = self._wall_cache[wall_name].mat_info
                if wall_mat_info:
                    wall_visual_geom.material = wall_mat_info.name

            mesh_colliders = make_wall_colliders(updated_wall_mesh, hole_type="door")
            for idx, mesh in enumerate(mesh_colliders):
                wall_collider_name = f"{wall_name}_collision_{idx}"
                wall_mesh_name = f"{wall_name}_collision_mesh_{idx}"
                mesh_collider_path = wall_mesh_filepath.parent / f"{wall_mesh_name}.obj"
                with stdout_redirected():
                    o3d.io.write_triangle_mesh(mesh_collider_path, mesh)
                    self.spec.add_mesh(
                        name=wall_mesh_name,
                        file=mesh_collider_path.relative_to(self.cfg.target_dir).as_posix(),
                        scale=[1, 1, -1],
                    )
                extra_wall_geom = body_handle.add_geom(
                    name=wall_collider_name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=wall_mesh_name,
                    quat=R.from_matrix(T_TRANSFORM[:3, :3]).as_quat(scalar_first=True),
                )
                set_defaults(
                    extra_wall_geom, self.spec.find_default(STRUCTURAL_WALL_CLASS), main_def
                )

                self._wall_cache[wall_name].mesh_colliders_paths.append(mesh_collider_path)

            if direction == 1 and asset_pos is not None:
                pos = unity_to_mj_pos(asset_pos)  # type: ignore

        assert pos is not None, f"Something went wrong when creating the door with id: {door_id}"

        p0 = wall0_info.vertices[0]
        p1 = wall0_info.vertices[1]
        direction = p0 - p1
        direction /= np.linalg.norm(direction)
        theta = np.rad2deg(np.arccos(np.dot(direction, [1, 0, 0])))
        if flip_2d_coord:
            theta += 180

        y_axis_rotation = R.from_euler("y", theta, degrees=True)
        curr_rot = R.from_matrix(T_FIX_ROTMAT)
        new_rot = y_axis_rotation * curr_rot
        quat = new_rot.as_quat()

        object_info = self.add_object_from_mjcf_model(door_json, position=pos, quaternion=quat)

        if object_info is None:
            raise ValueError(f"Couldn't add door mjcf '{door_id}'")

        self.cfg.counts[SceneObjectType.DOOR] += 1

        door_root_body = self.spec.body(object_info.hash_name)
        if door_root_body is not None:
            for geom in door_root_body.geoms:
                assert isinstance(geom, mj.MjsGeom)
                if geom.classname.name in (STRUCTURAL_CLASS, DYNAMIC_CLASS):
                    geom.density = 400.0

            if door_json.get("openable", False):
                if door_json["openness"] > 0:

                    def collect_joints(
                        body_spec: mj.MjsBody, joints_out: list[mj.MjsJoint]
                    ) -> None:
                        for joint in body_spec.joints:
                            assert isinstance(joint, mj.MjsJoint)
                            if joint.type in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}:
                                joints_out.append(joint)
                        for child_body in body_spec.bodies:
                            collect_joints(child_body, joints_out)

                    door_joints: list[mj.MjsJoint] = []
                    collect_joints(door_root_body, door_joints)
                    for joint in door_joints:
                        assert isinstance(joint, mj.MjsJoint)
                        has_handle = "handle" in joint.name or "handle" in joint.parent.name
                        if has_handle or joint.axis[2] != 0:
                            continue

                        before_rot = R.from_quat(joint.parent.quat, scalar_first=True)

                        axis = joint.axis
                        if axis[1] > 0:
                            y_axis_rotation = R.from_euler(
                                "y", door_json["openness"] * 90.0, degrees=True
                            )
                        else:
                            y_axis_rotation = R.from_euler(
                                "y", -door_json["openness"] * 90.0, degrees=True
                            )

                        after_rot = y_axis_rotation * before_rot
                        joint.parent.quat = after_rot.as_quat(scalar_first=True)
                        new_ref = (np.pi / 2.0) * door_json["openness"]

                        joint.ref = new_ref

            if not self.ignore_door_kinematic_flag or not door_json.get("openable", False):

                def remove_door_joints(body_spec: mj.MjsBody):
                    assert self.spec is not None
                    for joint in body_spec.joints:
                        self.spec.delete(joint)
                    for child_body in body_spec.bodies:
                        remove_door_joints(child_body)

                remove_door_joints(door_root_body)

    def add_window(self, window_json: dict[str, Any]) -> None:
        window_id = window_json.get("id", "")
        if window_id == "":
            msg = "Tried to process a window that didn't have an 'id'"
            self._logs["errors"].append(msg)
            log.error(msg)
            return

        _, wall0_info = self._get_cached_wall_info(window_json["wall0"])
        _, wall1_info = self._get_cached_wall_info(window_json["wall1"])

        _, room0_info = self._get_cached_room_info(window_json["room0"])
        _, room1_info = self._get_cached_room_info(window_json["room1"])

        flip_2d_coord = should_flip_asset_direction(
            room0_info.vertices, room1_info.vertices, wall0_info.vertices
        )

        hole_topleft_bottomright = np.array(
            [[xyz[k] for k in "xyz"] for xyz in window_json.get("holePolygon", [])]
        )
        pos: np.ndarray | None = None
        for direction, wall_info in zip((1, -1), (wall0_info, wall1_info), strict=True):
            wall_mesh = wall_info.mesh
            wall_mesh_filename = wall_info.filepath

            # Make hole in a wall and update the wall mesh
            updated_wall_mesh, asset_pos, _ = make_hole_in_wall(
                wall_mesh=wall_mesh,
                hole_topleft=hole_topleft_bottomright[0],
                hole_bottomright=hole_topleft_bottomright[1],
                direction=direction,
                asset_position_2d=[
                    window_json["assetPosition"]["x"],
                    window_json["assetPosition"]["y"],
                ],
                type="window",
                flip_2d_coord=flip_2d_coord,
            )
            with stdout_redirected():
                o3d.io.write_triangle_mesh(wall_mesh_filename.as_posix(), updated_wall_mesh)

            if direction == 1 and asset_pos is not None:
                pos = unity_to_mj_pos(asset_pos)  # type: ignore

        assert pos is not None, (
            f"Something went wrong when creating the window with id: {window_id}"
        )

        p0, p1 = wall0_info.vertices[0], wall0_info.vertices[1]
        direction = p0 - p1
        direction /= np.linalg.norm(direction)
        theta = np.rad2deg(np.arccos(np.dot(direction, [1, 0, 0])))

        if flip_2d_coord:
            theta += 180
            theta *= -1

        y_axis_rotation = R.from_euler("y", theta, degrees=True)
        curr_rot = R.from_matrix(T_FIX_ROTMAT)
        new_rot = y_axis_rotation * curr_rot
        quat = new_rot.as_quat()

        object_info = self.add_object_from_mjcf_model(window_json, position=pos, quaternion=quat)
        if object_info is None:
            raise ValueError(f"Couldn't add window mjcf '{window_id}'")

        self.cfg.counts[SceneObjectType.WINDOW] += 1

    def _get_cached_room_info(self, room_orig_name: str) -> tuple[str, RoomCachedInfo]:
        if room_orig_name not in self._name_mapping.rooms:
            raise ValueError(
                f"Tried to get room='{room_orig_name}' from names cache, but not found"
            )
        room_scene_name = self._name_mapping.rooms[room_orig_name]
        if room_scene_name not in self._room_cache:
            raise ValueError(
                f"Room with name: '{room_scene_name}' not found in rooms cache. Have you called 'add_room' first?"
            )
        return room_scene_name, self._room_cache[room_scene_name]

    def _get_cached_wall_info(self, wall_orig_name: str) -> tuple[str, WallCachedInfo]:
        if wall_orig_name not in self._name_mapping.walls:
            raise ValueError(
                f"Tried to get wall='{wall_orig_name}' from names cache, but not found"
            )
        wall_scene_name = self._name_mapping.walls[wall_orig_name]
        if wall_scene_name not in self._wall_cache:
            raise ValueError(
                f"Wall with name: '{wall_scene_name}' not found in walls cache. Have you called 'add_wall' first?"
            )
        return wall_scene_name, self._wall_cache[wall_scene_name]

    def _setup_material(
        self, obj_json: dict[str, Any], scene_obj_type: SceneObjectType
    ) -> MaterialInfo | None:
        assert self.spec is not None
        assert self.cfg.target_dir is not None

        mat_json: dict[str, Any] = {}
        match scene_obj_type:
            case SceneObjectType.WALL:
                mat_json = obj_json.get("material", {})
            case SceneObjectType.ROOM:
                mat_json = obj_json.get("floorMaterial", {})
            case _:
                log.warning(f"Tried setting up material for type '{scene_obj_type.name}'")

        mat_id: str = mat_json.get("name", "")
        mat_info: MaterialInfo | None = None
        if mat_id in self._assets_cache.materials:  # Material info found in thor materials
            mat_info = self._assets_cache.materials[mat_id]
        elif mat_id != "" and "color" in mat_json:  # This might be a custom material
            color_rgb = [mat_json["color"][c_id] for c_id in "rgb"]
            color = np.concatenate([color_rgb, [1.0]])
            mat_info = MaterialInfo(name=mat_id, rgba=color)
            self._assets_cache.materials[mat_id] = mat_info

        if (
            mat_info
            and self.spec.material(mat_id) is None
            and mat_id not in self._materials_added_set
        ):
            mat_spec = self.spec.add_material(name=mat_info.name)
            self._materials_added_set.add(mat_id)
            if scene_obj_type == SceneObjectType.WALL and "color" in mat_json:
                color_rgb = [mat_json["color"][c_id] for c_id in "rgb"]
                color = np.concatenate([color_rgb, [1.0]])
                mat_spec.rgba = color
            elif mat_info.rgba is not None:
                mat_spec.rgba = mat_info.rgba

            if mat_info.texture is not None:
                mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB.value] = mat_info.texture.name
                if (
                    mat_info.texture.name not in self._assets_cache.textures
                    and mat_info.texture.name not in self._textures_added_set
                ):
                    if self.spec.texture(mat_info.texture.name) is not None:
                        msg = f"Texture '{mat_info.texture.name}' is already in the spec"
                        self._logs["warnings"].append(msg)
                        log.warning(msg)
                    else:
                        self.spec.add_texture(
                            name=mat_info.texture.name,
                            file=mat_info.texture.file.as_posix(),
                            type=mj.mjtTexture.mjTEXTURE_2D,
                        )
                        self._textures_added_set.add(mat_info.texture.name)

        return mat_info

    def create_uv_coordinates(self, filepath: Path, uv_scale_factor: float = 1.0) -> None:
        if not filepath.is_file():
            return

        with stdout_redirected():
            bpy.context.scene.frame_set(1)
            bpy.ops.object.mode_set(mode="OBJECT")

            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete()

            bpy.ops.wm.obj_import(filepath=filepath.as_posix())

            # Apply UV mapping ------------------------------------
            for obj in bpy.data.objects:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                for modifier in obj.modifiers:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.parent_clear(type="CLEAR")
            # -----------------------------------------------------
            obj = bpy.context.selected_objects[0]

            bpy.context.view_layer.objects.active = obj

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")

            bpy.ops.mesh.remove_doubles()

            bpy.ops.mesh.normals_make_consistent(inside=False)

            bpy.ops.mesh.select_non_manifold()
            bpy.ops.mesh.remove_doubles()

            bpy.ops.mesh.uv_texture_add()

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.context.view_layer.update()

            bpy.ops.uv.cube_project(cube_size=uv_scale_factor)

            bpy.ops.object.mode_set(mode="OBJECT")
            obj.select_set(True)

            bpy.ops.wm.obj_export(
                filepath=filepath.as_posix(),
                export_selected_objects=True,
                export_uv=True,
                export_materials=False,
            )
