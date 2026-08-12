import gzip
import hashlib
import logging
import math
import os
import pickle
import re
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

import compress_json
import cv2
import msgpack
import mujoco as mj
import numpy as np
import open3d as o3d
import prior
from prior.utils.types import Dataset

from molmo_spaces.housegen.constants import (
    ARTICULABLE_DYNAMIC_CLASS,
    ASSETS_FROZEN_IN_SPACE,
    CATEGORIES_FROZEN_IN_SPACE,
    DYNAMIC_CLASS,
    STRUCTURAL_CLASS,
    TYPES_TO_USE_ONLY_PRIM,
    VISUAL_CLASS,
)

Vec3 = TypeVar("Vec3", list[float], tuple[float, float, float], np.ndarray)
Vec4 = TypeVar("Vec4", list[float], tuple[float, float, float, float], np.ndarray)

NP_ORIGIN = np.array([0.0, 0.0, 0.0])
NP_WORLD_UP = np.array([0.0, 0.0, 1.0])

VALID_SPLITS = {"train", "val", "test"}

THRESHOLD_EQUAL = 1e-10

NUM_VERTICES_SMALL_POLY = 20

JOINT_DEFAULT_DAMPING = 0.1
JOINT_DEFAULT_FRICTIONLOSS = 0.5
JOINT_DEFAULT_ARMATURE = 0
JOINT_DEFAULT_LIMITED = True

DEFAULT_SITES_GROUP = 5

HOLE_WALL_MARGIN = 0.05


class SceneObjectType(str, Enum):
    GENERIC = "generic"
    WALL = "wall"
    ROOM = "room"
    DOOR = "door"
    WINDOW = "window"
    THOR_OBJ = "thor_obj"
    OBJAVERSE_OBJ = "objaverse_obj"  # type associated with objaverse assets
    CUSTOM_OBJ = "custom_obj"  # type associated to custom geometry from iTHOR houses


@dataclass
class NameMapping:
    walls: dict[str, str] = field(default_factory=dict)
    rooms: dict[str, str] = field(default_factory=dict)


@dataclass
class ObjectNameMapping:
    bodies: dict[str, str] = field(default_factory=dict)
    joints: dict[str, str] = field(default_factory=dict)
    # geoms: dict[str, str] = field(default_factory=dict)
    sites: dict[str, str] = field(default_factory=dict)


@dataclass
class SceneObjectInfo:
    hash_name: str
    asset_id: str
    object_id: str
    category: str
    object_enum: SceneObjectType
    is_static: bool
    name_map: ObjectNameMapping
    mjcf_path: str = ""
    parent: str = ""
    room_id: int = 0
    children: list[str] = field(default_factory=list)


@dataclass
class SceneInfo:
    objects: dict[str, SceneObjectInfo] = field(default_factory=dict)


def unity_to_mj_pos(pos: Vec3) -> Vec3:
    assert len(pos) == 3, "Given vector should be of length 3"  # noqa: PLR2004

    if isinstance(pos, tuple):
        return (pos[0], pos[2], pos[1])
    elif isinstance(pos, list):
        return [pos[0], pos[2], pos[1]]
    elif isinstance(pos, np.ndarray):
        return np.array([pos[0], pos[2], pos[1]], dtype=pos.dtype)


# Helper used to suppress logs from bpy, taken from here:
# https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different
            _redirect_stdout(to=old_stdout)


def random_rgba() -> np.ndarray:
    return np.array(
        [np.random.random(), np.random.random(), np.random.random(), 1.0], dtype=np.float32
    )


def is_frozen_in_space(category: str | None, asset_id: str) -> bool:
    asset_id = asset_id.lower()
    return category in CATEGORIES_FROZEN_IN_SPACE or any(
        [asset_id.startswith(asset_id_start) for asset_id_start in ASSETS_FROZEN_IN_SPACE]
    )


# TODO(wilbert): might be better to use 'asset_id' instead of 'object_id'. For example, the
# object_id for garbage bins is 'bin_xyz' for iTHOR scenes, and 'GarbageCan|xyz' for ProcTHOR-10k


# TODO(wilbert): there are a few items that we could add/remove from the list:
# - Television: add as 'television' in the list
# - Doorway: add as 'door' in the list. This one is tricky bc prim requires to repair its colliders
def must_use_prim(object_id: str) -> bool:
    object_id_lower = object_id.lower()
    return any([ignore_type in object_id_lower for ignore_type in TYPES_TO_USE_ONLY_PRIM])


def extract_32_char_hex(filename: str) -> str | None:
    """Cuts the hash value from an objaverse assets id if applicable"""
    pattern = r"[0-9a-f]{32}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def set_defaults(
    obj: mj.MjsGeom | mj.MjsBody | mj.MjsJoint,
    class_def: mj.MjsDefault,
    main_def: mj.MjsDefault,
) -> None:
    if isinstance(obj, mj.MjsGeom):
        obj.classname = class_def
        obj.contype = class_def.geom.contype
        obj.conaffinity = class_def.geom.conaffinity
        obj.group = class_def.geom.group
        # NOTE(wilbert): density it setup in the model itself. Unfortunately mjspec doesn't have a
        # way to know if has some assigned density, so we will just check if close to the default
        if abs(obj.density - main_def.geom.density) < THRESHOLD_EQUAL:
            obj.density = class_def.geom.density
        if not math.isnan(class_def.geom.mass) and math.isnan(obj.mass):
            obj.mass = class_def.geom.mass

        obj.friction = class_def.geom.friction.copy()
        # TODO(wilbert): by commenting this out we avoid overwriting the solimp parameters from
        # the defaults of each default class. Have to double check that we didn't change the
        # behavior on other objects

        # obj.solimp = class_def.geom.solimp.copy()
        obj.solref = class_def.geom.solref.copy()
    elif isinstance(obj, mj.MjsBody):
        pass
    elif isinstance(obj, mj.MjsJoint):
        pass


def load_objaverse_houses(
    house_dataset_path: Path,
    split: Literal["train", "val", "test"] = "train",
    max_houses: int = int(1e6),
) -> Dataset:
    max_houses_per_split = dict(train=0, val=0, test=0)
    max_houses_per_split[split] = max_houses

    return prior.load_dataset(
        "procthor-objaverse-internal",
        revision="local",
        path_to_splits=None,
        split_to_path={
            split: (house_dataset_path / f"{split}.jsonl.gz").as_posix() for split in VALID_SPLITS
        },
        max_houses_per_split=max_houses_per_split,
    )[split]


def load_holodeck_houses(
    house_dataset_path: Path,
    split: Literal["train", "val", "test"] = "train",
    max_houses: int = int(1e6),
):
    max_houses_per_split = dict(train=0, val=0, test=0)
    max_houses_per_split[split] = max_houses

    return prior.load_dataset(
        "procthor-objaverse-internal",
        revision="local",
        path_to_splits=None,
        split_to_path={
            split: (house_dataset_path / f"{split}.jsonl.gz").as_posix() for split in VALID_SPLITS
        },
        max_houses_per_split=max_houses_per_split,
    )[split]


def load_objaverse_object(objaverse_id: str, objaverse_dir: Path) -> dict:
    datapath_json = objaverse_dir / objaverse_id / f"{objaverse_id}.json"
    datapath_pkl = objaverse_dir / objaverse_id / f"{objaverse_id}.pkl.gz"
    datapath_msgpack = objaverse_dir / objaverse_id / f"{objaverse_id}.msgpack.gz"
    has_data_file = datapath_json.is_file() or datapath_pkl.is_file() or datapath_msgpack.is_file()

    if not has_data_file:
        raise FileNotFoundError(
            f"Object with id {objaverse_id} doesn't have valid data in any supported format"
        )

    object_json = {}
    if datapath_pkl.is_file():
        with gzip.open(datapath_pkl, "rb") as fhandle:
            object_json = pickle.load(fhandle)
    elif datapath_msgpack.is_file():
        with gzip.open(datapath_msgpack, "rb") as fhandle:
            object_json = msgpack.unpackb(fhandle.read(), raw=False)
    elif datapath_json.is_file():
        object_json = compress_json.load(datapath_json.as_posix())
    return object_json


def create_mesh_from_objaverse_data(
    filepath: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    texture: Any | None = None,
    is_collider: bool = False,
    rotate_yaxis: float | None = 0,
    rotate_zaxis: float | None = 0,
    translate_center: bool = False,
    translate_offset: np.ndarray | None = None,
    overwrite_if_exists: bool = True,
) -> tuple[bool, mj.mjtMeshInertia, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    is_watertight = mesh.is_watertight()
    inertia_mode = mj.mjtMeshInertia.mjMESH_INERTIA_LEGACY
    if is_watertight:
        inertia_mode = (
            mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
            if mesh.get_volume() <= 1e-14  # noqa: PLR2004
            else mj.mjtMeshInertia.mjMESH_INERTIA_LEGACY
        )
    else:
        inertia_mode = (
            mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
            if len(mesh.vertices) < 4  # noqa: PLR2004
            else mj.mjtMeshInertia.mjMESH_INERTIA_LEGACY
        )

    if is_collider and inertia_mode == mj.mjtMeshInertia.mjMESH_INERTIA_SHELL:
        return False, inertia_mode, center

    if translate_center:
        mesh.translate(-center)
    elif translate_offset is not None:
        mesh.translate(translate_offset)

    if normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    if uvs is not None:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    if texture:
        mesh.textures = [o3d.geometry.Image(texture)]

    # # NOTE(wilbert): don't apply rotations as it seems the vertex data is already rotated
    # if rotate_yaxis is not None:
    #     rotate_obb_matrix = R.from_rotvec([0.0, rotate_yaxis, 0.0], degrees=True).as_matrix()
    #     mesh.rotate(rotate_obb_matrix, NP_ORIGIN)
    # if rotate_zaxis is not None:
    #     rotate_obb_matrix = R.from_rotvec([0.0, 0.0, rotate_zaxis], degrees=False).as_matrix()
    #     mesh.rotate(rotate_obb_matrix, NP_ORIGIN)

    if is_collider and not mesh.is_watertight():
        return False, mj.mjtMeshInertia.mjMESH_INERTIA_SHELL, center

    if not filepath.exists() or overwrite_if_exists:
        o3d.io.write_triangle_mesh(filepath.as_posix(), mesh)

    return True, inertia_mode, center


def create_mujoco_model_from_objaverse(
    objaverse_id: str,
    objaverse_dir: Path,
    save_folder: Path,
    name: str | None = None,
    copy_extras: bool = True,
) -> None:
    save_folder.mkdir(exist_ok=True)

    object_json = load_objaverse_object(objaverse_id, objaverse_dir)

    spec: mj.MjSpec = mj.MjSpec()
    spec.modelname = name if name is not None else objaverse_id
    dummy_default = spec.add_default("/", spec.default)

    visual_default = spec.add_default(VISUAL_CLASS, dummy_default)
    visual_default.geom.contype = 0
    visual_default.geom.conaffinity = 0
    visual_default.geom.group = 0
    visual_default.geom.mass = 1e-8

    structural_default = spec.add_default(STRUCTURAL_CLASS, dummy_default)
    structural_default.geom.contype = 8
    structural_default.geom.conaffinity = 15
    structural_default.geom.group = 4
    structural_default.geom.density = 500
    structural_default.geom.friction = np.array([0.9, 0.9, 0.001], dtype=np.float64)
    structural_default.geom.solref = np.array([0.025, 1], dtype=np.float64)
    structural_default.geom.solimp[:3] = np.array([0.998, 0.998, 0.001], dtype=np.float64)

    dynamic_default = spec.add_default(DYNAMIC_CLASS, dummy_default)
    dynamic_default.geom.contype = 1
    dynamic_default.geom.conaffinity = 15
    dynamic_default.geom.group = 4
    dynamic_default.geom.density = 500
    dynamic_default.geom.friction = np.array([0.9, 0.9, 0.001], dtype=np.float64)
    dynamic_default.geom.solref = np.array([0.025, 1], dtype=np.float64)
    dynamic_default.geom.solimp[:3] = np.array([0.998, 0.998, 0.001], dtype=np.float64)

    art_dyn_default = spec.add_default(ARTICULABLE_DYNAMIC_CLASS, dummy_default)
    art_dyn_default.geom.contype = 0
    art_dyn_default.geom.conaffinity = 7
    art_dyn_default.geom.group = 4
    art_dyn_default.geom.density = 500
    art_dyn_default.geom.friction = np.array([0.9, 0.9, 0.001], dtype=np.float64)
    art_dyn_default.geom.solref = np.array([0.025, 1], dtype=np.float64)
    art_dyn_default.geom.solimp[:3] = np.array([0.998, 0.998, 0.001], dtype=np.float64)

    root_body: mj.MjsBody = spec.worldbody.add_body(name=objaverse_id)
    root_body.add_joint(
        name=f"{objaverse_id}_jntfree",
        type=mj.mjtJoint.mjJNT_FREE,
        damping=JOINT_DEFAULT_DAMPING,
    )

    pose_z_rot_angle = 0.0
    asset_scale = [-1.0, 1.0, 1.0]
    annotations_path = objaverse_dir / objaverse_id / "annotations.json.gz"
    if annotations_path.exists():
        annotations = compress_json.load(annotations_path.as_posix())
        # NOTE(wilbert): dont' use the scale, as it seems the object has already been scaled with bpy
        # annotation_scale = annotations.get("scale", 1.0)
        # asset_scale = [-annotation_scale, annotation_scale, annotation_scale]
        pose_z_rot_angle = annotations.get("pose_z_rot_angle", 0.0)

    # Make main visual object ----------------------------------------------------------------------
    vertices = np.array([[xyz["x"], xyz["y"], xyz["z"]] for xyz in object_json["vertices"]])
    triangles = np.array(object_json["triangles"]).reshape(-1, 3)
    normals = np.array([[xyz["x"], xyz["y"], xyz["z"]] for xyz in object_json["normals"]])
    uvs = np.array([[xy["x"], 1.0 - xy["y"]] for xy in object_json["uvs"]])

    texture = None
    if "albedoTexturePath" in object_json:
        albedo_path = objaverse_dir / objaverse_id / object_json["albedoTexturePath"]
        texture = o3d.io.read_image(albedo_path.as_posix())

    mesh_visual_name = Path(f"{objaverse_id}_visual.obj")
    mesh_created, inertia_mode, center = create_mesh_from_objaverse_data(
        save_folder / mesh_visual_name,
        vertices=vertices,
        triangles=triangles,
        normals=normals,
        uvs=uvs,
        texture=texture,
        is_collider=False,
        rotate_yaxis=object_json.get("yRotOffset"),
        rotate_zaxis=pose_z_rot_angle,
        translate_center=True,
    )

    if not mesh_created:
        raise ValueError(f"Couldn't create a valid visual geom for object {objaverse_id} ")

    # We're assumming that we will copy the required elements to the save folder, and that it will
    # be located at the same level as the generated mjcf xml

    vis_material_name = f"{objaverse_id}_mat"

    spec.add_mesh(
        name=f"{objaverse_id}_visual_mesh",
        file=mesh_visual_name.name,
        scale=asset_scale,
        inertia=inertia_mode,
    )

    if texture:
        vis_texture_name = f"{objaverse_id}_albedo"
        spec.add_texture(
            name=vis_texture_name,
            file=f"{mesh_visual_name.stem}_0.png",
            type=mj.mjtTexture.mjTEXTURE_2D,
        )
        mat_spec = spec.add_material(name=vis_material_name)
        mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = vis_texture_name  # type: ignore

    visual_geom = root_body.add_geom(
        name=f"{objaverse_id}_visual",
        type=mj.mjtGeom.mjGEOM_MESH,
        meshname=f"{objaverse_id}_visual_mesh",
    )
    visual_geom.classname = visual_default
    # TODO(wilbert): setting classname doesn't propagate the defaults when using mjspec
    visual_geom.contype = visual_default.geom.contype
    visual_geom.conaffinity = visual_default.geom.conaffinity
    visual_geom.group = visual_default.geom.group
    visual_geom.mass = visual_default.geom.mass

    if texture:
        visual_geom.material = vis_material_name

    # ----------------------------------------------------------------------------------------------

    # Generate colliders ---------------------------------------------------------------------------
    n_colliders = 0
    for i, col_data in enumerate(object_json.get("colliders", [])):
        mesh_col_name = Path(f"{objaverse_id}_collider{i}.obj")
        c_vertices = np.array([[xyz["x"], xyz["y"], xyz["z"]] for xyz in col_data["vertices"]])
        c_triangles = np.array(col_data["triangles"]).reshape(-1, 3)

        mesh_created, inertia_mode, _ = create_mesh_from_objaverse_data(
            save_folder / mesh_col_name.name,
            vertices=c_vertices,
            triangles=c_triangles,
            is_collider=True,
            rotate_yaxis=object_json.get("yRotOffset"),
            rotate_zaxis=pose_z_rot_angle,
            translate_center=False,
            translate_offset=-center,
        )

        if not mesh_created:
            continue

        n_colliders += 1

        spec.add_mesh(
            name=mesh_col_name.stem,
            file=mesh_col_name.name,
            scale=asset_scale,
            inertia=inertia_mode,
        )

        collider_geom = root_body.add_geom(
            name=mesh_col_name.stem,
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=mesh_col_name.stem,
            rgba=random_rgba(),
        )
        collider_geom.classname = dynamic_default
        # TODO(wilbert): setting classname doesn't propagate the defaults when using mjspec
        collider_geom.contype = dynamic_default.geom.contype
        collider_geom.conaffinity = dynamic_default.geom.conaffinity
        collider_geom.contype = dynamic_default.geom.contype
        collider_geom.conaffinity = dynamic_default.geom.conaffinity
        collider_geom.group = dynamic_default.geom.group
        collider_geom.density = dynamic_default.geom.density
        collider_geom.friction = dynamic_default.geom.friction
        collider_geom.solref = dynamic_default.geom.solref
        collider_geom.solimp = dynamic_default.geom.solimp

    if n_colliders < 1:
        raise ValueError(f"Couldn't create any colliders for objaverse item {objaverse_id}")

    # ----------------------------------------------------------------------------------------------

    # Generate the mjcf model ----------------------------------------------------------------------
    workdir = Path.cwd()
    os.chdir(save_folder)
    _ = spec.compile()
    with open(f"{objaverse_id}.xml", "w") as fhandle:
        fhandle.write(spec.to_xml())
    os.chdir(workdir)
    # ----------------------------------------------------------------------------------------------

    # Copy other data files that we might need from the original folder ----------------------------
    if copy_extras:

        def copy_item_to_mjcf_folder(item_name: str) -> None:
            src_path = objaverse_dir / objaverse_id / item_name
            dst_path = save_folder / item_name
            if src_path.exists():
                if src_path.is_file():
                    shutil.copy(src_path, dst_path)
                elif src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        items_to_copy = (
            "annotations.json.gz",
            "blender_renders",
            "emission.jpg",
            "metallic_smoothness.jpg",
            "normal.jpg",
            "thor_metadata.json",
            "thor_renders",
        )
        for item in items_to_copy:
            copy_item_to_mjcf_folder(item)
    # ----------------------------------------------------------------------------------------------


def compute_normal_for_face(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    p_0 = vertices[face[0]]
    p_1 = vertices[face[1]]
    p_2 = vertices[face[2]]
    f_normal = np.cross(p_1 - p_0, p_2 - p_0)
    f_normal /= np.linalg.norm(f_normal)
    return f_normal


def create_wall_mesh(
    vertices: np.ndarray, thickness: float = 0.005, dir: int = 1
) -> o3d.geometry.TriangleMesh:
    # Flat surface
    # 0 1
    # 2 3
    triangles = [[0, 1, 2], [1, 3, 2]]

    # Calculate the normal of the plane
    # noinspection PyUnreachableCode
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normal /= np.linalg.norm(normal)

    # Extrude the flat mesh with a thickness 't'
    t = thickness  # / 2.0
    vertices_extruded = vertices + t * normal * dir
    triangles_extruded = [[4, 5, 6], [5, 7, 6]]
    flipped_triangles = [list(reversed(triangle)) for triangle in triangles_extruded]

    # Move the original vertices to the opposite direction
    # vertices = vertices - t * normal

    # Create new triangles for the sides
    triangles_sides = [
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 7],
        [1, 7, 3],
        [3, 7, 6],
        [3, 6, 2],
        [2, 6, 4],
        [2, 4, 0],
    ]

    # Combine the triangles from the flat part and the sides
    triangles_all = []
    triangles_all.extend(triangles_sides)
    triangles_all.extend(triangles)
    triangles_all.extend(flipped_triangles)
    if dir > 0:
        triangles_all = [list(reversed(triangle)) for triangle in triangles_all]

    all_vertices = []
    all_vertices.append(vertices)
    all_vertices.append(vertices_extruded)

    # Create a new mesh for the extruded part
    mesh_extruded = o3d.geometry.TriangleMesh()
    mesh_extruded.vertices = o3d.utility.Vector3dVector(np.concatenate(all_vertices))
    mesh_extruded.triangles = o3d.utility.Vector3iVector(np.array(triangles_all))

    return mesh_extruded


def create_room_mesh(vertices: np.ndarray, thickness: float = 0.005) -> o3d.geometry.TriangleMesh:
    def is_convex(p1, p2, p3):
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return cross_product < 0

    def is_ear(p1, p2, p3, polygon, remaining_indices) -> bool:
        # First check if the triangle is convex
        if not is_convex(p1, p2, p3):
            return False

        triangle_edges = [(p1, p2), (p2, p3), (p3, p1)]

        # Check for intersections with polygon edges
        for edge in triangle_edges:
            edge_start, edge_end = edge
            for i in range(len(polygon)):
                poly_edge_start = polygon[i]
                poly_edge_end = polygon[(i + 1) % len(polygon)]

                if do_lines_intersect(
                    edge_start[:2],
                    edge_end[:2],
                    poly_edge_start[:2],
                    poly_edge_end[:2],
                ):
                    return False

        # Check if any remaining points are inside the triangle
        for idx in remaining_indices:
            if idx not in [p1[2], p2[2], p3[2]]:
                test_point = polygon[idx]
                if is_point_inside_triangle(p1, p2, p3, test_point):
                    return False

        return True

    def is_point_inside_triangle(p1, p2, p3, test_point):
        # Check if the test point is inside the triangle formed by p1, p2, and p3
        b1 = sign(test_point, p1, p2) < 0.0
        b2 = sign(test_point, p2, p3) < 0.0
        b3 = sign(test_point, p3, p1) < 0.0
        return b1 == b2 == b3

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def do_lines_intersect(p1, p2, q1, q2) -> bool:
        def determinant(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        def subtract(v1, v2):
            return (v1[0] - v2[0], v1[1] - v2[1])

        n = subtract(p2, p1)
        m = subtract(q1, q2)
        p = subtract(q1, p1)

        D = determinant(n, m)
        Qx = determinant(p, m)
        Qy = determinant(n, p)

        if D == 0:
            # Segments are parallel
            if Qx != 0 or Qy != 0:
                return False  # Segments are parallel but not collinear
            # Segments are collinear, check for overlap
            if p1 != q1 and p1 != q2 and p2 != q1 and p2 != q2:
                return False  # Segments are collinear and overlap
            return False  # Segments are collinear but disjoint

        t = Qx / D
        s = Qy / D

        segments_intersect = 0 < t < 1 and 0 < s < 1
        return segments_intersect
        # True  # Segments intersect
        # return False  # Segments do not intersect

    def ear_clip_triangulation(polygon):
        n = len(polygon)
        if n < 3:
            return []

        triangles = []
        remaining_indices = list(range(n))

        while len(remaining_indices) >= 3:
            ear_found = False
            best_ear: tuple | None = None
            min_area = float("inf")  # Use area as a metric to choose the smallest valid triangle

            for i in range(len(remaining_indices)):
                idx = remaining_indices[i]
                prev_idx = remaining_indices[i - 1]
                next_idx = remaining_indices[(i + 1) % len(remaining_indices)]

                p1, p2, p3 = polygon[prev_idx], polygon[idx], polygon[next_idx]

                if is_ear(p1, p2, p3, polygon, remaining_indices):
                    # Calculate triangle area
                    area = (
                        abs((p2[0] - p1[0]) * (p3[2] - p1[2]) - (p3[0] - p1[0]) * (p2[2] - p1[2]))
                        / 2.0
                    )

                    # Check if this triangle would be better than our current best
                    if area < min_area:
                        min_area = area
                        best_ear = (i, [p1[2], p2[2], p3[2]])
                        ear_found = True

            if ear_found:
                assert best_ear is not None, "best_ear should be valid by now if found ear >.<"
                i, triangle = best_ear
                triangles.append(triangle)
                remaining_indices.pop(i)
            else:
                # If no ear is found, try to find any valid triangle
                for i in range(len(remaining_indices)):
                    for j in range(i + 1, len(remaining_indices)):
                        for k in range(j + 1, len(remaining_indices)):
                            p1 = polygon[remaining_indices[i]]
                            p2 = polygon[remaining_indices[j]]
                            p3 = polygon[remaining_indices[k]]

                            # Check if this forms a valid triangle
                            if is_convex(p1, p2, p3) and not any(
                                is_point_inside_triangle(p1, p2, p3, polygon[idx])
                                for idx in remaining_indices
                                if idx
                                not in [
                                    remaining_indices[i],
                                    remaining_indices[j],
                                    remaining_indices[k],
                                ]
                            ):
                                triangles.append([p1[2], p2[2], p3[2]])
                                # Remove middle vertex
                                remaining_indices.remove(remaining_indices[j])
                                ear_found = True
                                break
                        if ear_found:
                            break
                    if ear_found:
                        break

                if not ear_found:
                    break

        return triangles

    # ------------------------------------------------------------------------------------------
    # NOTE(wilbert): have to simplify as some polygons have colinear vertices, and that breaks
    # the logic below for the find_plane_normal function
    vertices = (
        cv2.approxPolyDP(vertices[:, [0, 2]].astype(np.float32), 1e-6, closed=True)
        .squeeze()
        .astype(vertices.dtype)
    )
    vertices = np.c_[vertices[:, 0], np.zeros(len(vertices)), vertices[:, 1]]
    # ------------------------------------------------------------------------------------------

    # NOTE(wilbert): This reordering worked only for quads, but collapses some type of polygons,
    # like room|3 in train_16 from procthor-10k train set. For now we're disabling it, as after a
    # quick check it seems that it's not required anymore

    # # Reorder vertices
    # if vertices[0][2] > vertices[1][2] and vertices[2][2] > vertices[3][2]:
    #     vertices = np.array([vertices[1], vertices[0], vertices[3], vertices[2]])

    triangles = ear_clip_triangulation(
        [(vertices[i][0], vertices[i][2], i) for i in range(len(vertices))]
    )

    # Create bottom mesh
    def find_plane_normal(vertices):
        # Ensure we have at least 3 points
        if len(vertices) < 3:
            raise ValueError("At least three points are required to define a plane.")

        # Try to find three non-collinear points
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                for k in range(j + 1, len(vertices)):
                    # Check for collinearity
                    # noinspection PyUnreachableCode
                    if not np.allclose(
                        np.cross(
                            np.array(vertices[j]) - np.array(vertices[i]),
                            np.array(vertices[k]) - np.array(vertices[i]),
                        ),
                        [0, 0, 0],
                    ):
                        # Form two vectors in the plane
                        v1 = np.array(vertices[j]) - np.array(vertices[i])
                        v2 = np.array(vertices[k]) - np.array(vertices[i])

                        # Calculate the cross product to get the normal vector
                        # noinspection PyUnreachableCode
                        normal = np.cross(v1, v2)

                        return normal / np.linalg.norm(normal)

        raise ValueError("All combinations of points are collinear. Cannot define a unique plane.")

    normal = find_plane_normal(vertices)
    if np.any(normal < 0):
        normal *= -1

    # Extrude the flat mesh with a thickness 't'
    vertices_extruded = vertices - thickness * normal * 10.0  # below XY plane
    triangles_extruded = np.array(triangles) + len(vertices)
    triangles_extruded = [list(reversed(triangle)) for triangle in triangles_extruded]

    # Create the sides of the mesh
    # 0 1  --- (n=4)
    # 4 5
    n = len(vertices)
    triangles_sides = []
    for i in range(n):
        triangles_sides.append([i, (i + 1) % n, i + n])
        triangles_sides.append([(i + 1) % n, (i + 1) % n + n, i + n])
    triangles_sides = [list(reversed(triangle)) for triangle in triangles_sides]

    # Combine the top, bottom, and sides to create the final mesh
    mesh = o3d.geometry.TriangleMesh()  # .create_box(width=1, height=1, depth=1)
    all_vertices = np.concatenate((vertices, vertices_extruded))
    all_triangles = np.concatenate((triangles, triangles_extruded, triangles_sides))
    uv_coordinates = all_vertices[:, :2]  # Assuming a simple planar mapping along the XY plane
    uv_coordinates = uv_coordinates - np.min(uv_coordinates, axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coordinates)
    return mesh


def generate_object_hash(asset_id: str) -> str:
    hasher = hashlib.md5()
    hasher.update(asset_id.encode())
    return hasher.hexdigest()


# --------------------------------------------------------------------------------------------------
# Helper functions used to build parts of the scene and some geometric stuff
# --------------------------------------------------------------------------------------------------


def is_point_in_polygon_winding_number(point, polygon, tol=1e-6):
    """
    Winding number algorithm - more robust for complex polygons and edge cases.
    """
    x, y = point
    n = len(polygon)
    winding_number = 0

    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]

        # Check if point is on the edge
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < tol and abs(dy) < tol:
            continue

        # Check if point is on this edge
        t = ((x - x0) * dx + (y - y0) * dy) / (dx**2 + dy**2)
        if 0 - tol <= t <= 1 + tol:
            closest_x = x0 + t * dx
            closest_y = y0 + t * dy
            if (x - closest_x) ** 2 + (y - closest_y) ** 2 < tol**2:
                return True

        # Winding number calculation using cross product
        # This is more robust than the previous method
        cross_product = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)

        # Upward crossing
        if y0 <= y and y1 > y:
            if cross_product > tol:  # Use tolerance for robustness
                winding_number += 1
        # Downward crossing
        elif y0 > y and y1 <= y:
            if cross_product < -tol:  # Use tolerance for robustness
                winding_number -= 1

    return winding_number != 0


def is_point_in_polygon(point, polygon, tol=1e-6):
    """
    Check if a point is inside a polygon using multiple algorithms for robustness.
    """
    x, y = point
    n = len(polygon)

    # Debug: print polygon info
    log.debug(f"Checking point {point} in polygon with {n} vertices")
    if n <= NUM_VERTICES_SMALL_POLY:  # Print vertices for smaller polygons
        log.debug(f"Polygon vertices: {polygon}")

    # Method 1: Try winding number first (more reliable for complex polygons)
    winding_result = is_point_in_polygon_winding_number(point, polygon, tol)
    log.debug(f"Winding number method says: {'inside' if winding_result else 'outside'}")

    # Method 2: Ray casting as backup
    inside = False
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]

        # Edge check - if point is on the edge, consider it inside
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < tol and abs(dy) < tol:
            continue  # Skip degenerate edges

        # Check if point is on this edge
        t = ((x - x0) * dx + (y - y0) * dy) / (dx**2 + dy**2)
        if 0 - tol <= t <= 1 + tol:
            closest_x = x0 + t * dx
            closest_y = y0 + t * dy
            if (x - closest_x) ** 2 + (y - closest_y) ** 2 < tol**2:
                log.debug("Point is on edge - returning True")
                return True

        # Ray-casting algorithm
        if (y0 > y) != (y1 > y):
            # Avoid division by zero
            if abs(y1 - y0) < tol:
                continue
            x_intersect = (x1 - x0) * (y - y0) / (y1 - y0) + x0

            # Use tolerance for intersection comparison
            if x < x_intersect - tol:
                inside = not inside
            elif abs(x - x_intersect) < tol:
                # Point is very close to the edge, consider it inside
                log.debug("Point is very close to edge - returning True")
                return True

    ray_casting_result = inside
    log.debug(f"Ray casting method says: {'inside' if ray_casting_result else 'outside'}")

    # Use winding number result as primary (more reliable)
    final_result = winding_result
    log.debug(f"Final result using winding number: {'inside' if final_result else 'outside'}")

    return final_result


def point_along_inward_normal(p0, p1, polygon, distance=0.1, tol=1e-6):
    """
    Get a point along the inward normal of a polygon edge, inside the polygon.

    Parameters:
        p0, p1 : tuple
            Endpoints of the polygon edge
        polygon : list of (x, y)
            Polygon vertices in order
        distance : float
            Distance along the inward normal
        tol : float
            Tolerance for point-in-polygon check

    Returns:
        np.array: Point inside polygon along inward normal
    """
    edge = np.array(p1) - np.array(p0)
    # Normal vector (perpendicular)
    normal = np.array([-edge[1], edge[0]])
    normal = normal / np.linalg.norm(normal)

    midpoint = (np.array(p0) + np.array(p1)) / 2
    test_point = midpoint + distance * normal  # tiny step to check inward

    # Check inward direction
    is_inside = is_point_in_polygon(test_point, polygon, tol)
    log.debug(f"First test: is_point_in_polygon({test_point}, polygon, {tol}) = {is_inside}")

    if not is_inside:
        normal = -normal  # flip inward
        test_point = midpoint + distance * normal  # tiny step to check inward
        log.debug(f"Flipped normal: {normal}")
        log.debug(f"New test_point: {test_point}")
        is_inside = is_point_in_polygon(test_point, polygon, tol)
        log.debug(f"Second test: is_point_in_polygon({test_point}, polygon, {tol}) = {is_inside}")

        if not is_inside:
            log.warning("Still not inside after flipping normal!")
            normal = -normal  # flip back

    # Return point along normal
    result = test_point  # midpoint + distance * normal
    log.debug(f"  Final result: {result}")
    return result


def should_flip_asset_direction(
    room0_vertices: np.ndarray, room1_vertices: np.ndarray, wall0_vertices: np.ndarray
) -> bool:
    """
    Determine if the asset holePolygon should be relative to left bottom or right bottom

    Args:
        room0_vertices: Vertices for the first room.
        room1_vertices: Vertices for the second room.
        wall0_vertices: Vertices for the wall.

    Returns:
        bool: True if the asset direction should be flipped, False otherwise.
    """

    room0 = np.mean(room0_vertices, axis=0)
    room1 = np.mean(room1_vertices, axis=0)
    room_diff = room1 - room0

    # wall noraml
    p0 = wall0_vertices[0]
    p1 = wall0_vertices[1]
    e1 = p1 - p0
    # e2 = np.array([0, 0, 1])  # up vector p2 - p0
    e2 = np.array([0, 1, 0])  # up vector p2 - p0
    face_normal = np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2))
    ind = np.nonzero(face_normal)
    assert len(ind) == 1
    ind = ind[0]

    # if wall is adjacent to the world, room0 == room1
    if np.sum(room_diff) == 0:
        mid_wall_dir = (p0 + p1) / 2  # - p0

        vertices = room0_vertices  # self.editor.object_id_to_metadata[room0_id]["mesh"].vertices
        vertices = [(v[0], v[2]) for v in vertices]
        _p0 = [p0[0], p0[2]]
        _p1 = [p1[0], p1[2]]
        room_point = point_along_inward_normal(_p0, _p1, vertices, tol=1e-6)
        room_points = [room_point[0], 0, room_point[1]]
        window_dir = (room_points - mid_wall_dir) / np.linalg.norm(room_points - mid_wall_dir)
        if ind == 0 and window_dir[0] < 0:
            return True
        return bool(ind == 2 and window_dir[2] > 0)

    window_dir = room_diff / np.linalg.norm(room_diff)
    # Unity X direction
    if ind == 0 and window_dir[0] > 0:
        return True
    # Unity Y direction
    return bool(ind == 2 and window_dir[2] < 0)


def make_hole_in_wall(
    wall_mesh: o3d.geometry.TriangleMesh,
    hole_topleft: np.ndarray,
    hole_bottomright: np.ndarray,
    direction: int = 1,
    asset_position_2d: Sequence[float] | None = None,
    type: str = "window",
    flip_2d_coord: bool = False,
    fix_problematic_holodeck: bool = False,
):
    """Need to make a visual and a set of convex collision mesh for the hole in the wall.
    3 collision meshes for the door
    4 collision meshes for the window
    """

    def convert_2d_to_3d(relative_2d_vertices, plane_origin, xvec, yvec):
        # Assuming the plane is the XY plane
        points = []
        for i in range(len(relative_2d_vertices)):
            point = (
                plane_origin + relative_2d_vertices[i][0] * xvec + relative_2d_vertices[i][1] * yvec
            )
            points.append(point)
        return np.array(points)  # np.stack([x,y,z], axis=1)

    # the two points are top left corner and bottom right corner x,y coordinates relative to the wall coordinate space (so 0,0 would be the bottom left point where the wall starts at the floor, the first point specified in the wall's polygon list
    wall_vertices = np.array(wall_mesh.vertices)  # 8 vertices

    # try:
    # noinspection PyUnreachableCode
    normal = np.cross(wall_vertices[1] - wall_vertices[0], wall_vertices[2] - wall_vertices[0])
    # except (ValueError, IndexError):
    #    pass
    normal /= np.linalg.norm(normal)

    def angle_between_vectors(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        cos_theta = dot_product / magnitude_product
        theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors
        theta_degrees = np.degrees(theta_radians)
        return theta_degrees

    asset_position_3d = None
    asset_rotation_3d = np.zeros(3)  # [0, y-axis, 0] in Unity
    asset_rotation_3d[1] = angle_between_vectors(np.array([1, 0, 0]), normal)

    # Updated top surface
    # 2       3
    #   10 11
    #   8 9
    # 0       1
    hole_vertices_2d = np.array(
        [
            [hole_topleft[0], hole_topleft[1]],
            [hole_bottomright[0], hole_topleft[1]],
            [hole_topleft[0], hole_bottomright[1]],
            [hole_bottomright[0], hole_bottomright[1]],
        ]
    )

    plane_origin_index = 0
    xvec = wall_vertices[1] - wall_vertices[0]
    yvec = wall_vertices[2] - wall_vertices[0]
    xvec = xvec / np.linalg.norm(xvec)
    yvec = yvec / np.linalg.norm(yvec)

    wall_length = np.sqrt(
        np.sum((wall_vertices[plane_origin_index + 1] - wall_vertices[plane_origin_index]) ** 2)
    )
    if flip_2d_coord:
        hole_vertices_2d[:, 0] = wall_length - hole_vertices_2d[:, 0]
        hole_vertices_2d = np.array(
            [
                hole_vertices_2d[1],
                hole_vertices_2d[0],
                hole_vertices_2d[3],
                hole_vertices_2d[2],
            ]
        )

    if asset_position_2d is not None:
        if flip_2d_coord:
            asset_position_2d = [
                wall_length - asset_position_2d[0],
                asset_position_2d[1],
            ]

        asset_position_3d = convert_2d_to_3d(
            [asset_position_2d], wall_vertices[plane_origin_index], xvec, yvec
        )[0]

    hole_vertices_top = convert_2d_to_3d(
        hole_vertices_2d, wall_vertices[plane_origin_index], xvec, yvec
    )
    hole_vertices_bottom = convert_2d_to_3d(
        hole_vertices_2d, wall_vertices[plane_origin_index + 4], xvec, yvec
    )

    # NOTE(wilbert): we have to do this step for holodeck houses, as some have doors right next to
    # the end of a wall, which makes it to have very thin meshes after making the whole, which is
    # breaking and giving segfaults when trying to export those houses
    if fix_problematic_holodeck:
        # Find the main axis that spans the wall
        main_idx = -1
        if np.allclose(hole_vertices_top[:, 0], hole_vertices_top[0, 0], atol=1e-5):
            main_idx = 0
        elif np.allclose(hole_vertices_top[:, 2], hole_vertices_top[0, 2], atol=1e-5):
            main_idx = 2

        # Check if we're too close to any of the borders of the wall
        if main_idx != -1:
            hole_base_too_close_0 = False
            hole_base_too_close_1 = False
            asset_pos_off_p0 = 0.0
            asset_pos_off_p1 = 0.0

            wall_idx = 2 if main_idx == 0 else 0
            wall_base_p0 = wall_vertices[0, wall_idx]
            wall_base_p1 = wall_vertices[1, wall_idx]
            hole_base_p0 = hole_vertices_top[0, wall_idx]
            hole_base_p1 = hole_vertices_top[1, wall_idx]

            hole_base_too_close_0 = abs(hole_base_p0 - wall_base_p0) < HOLE_WALL_MARGIN
            hole_base_too_close_1 = abs(wall_base_p1 - hole_base_p1) < HOLE_WALL_MARGIN

            if hole_base_too_close_0:
                hole_vertices_top[0, wall_idx] = wall_base_p0 + HOLE_WALL_MARGIN
                hole_vertices_top[2, wall_idx] = wall_base_p0 + HOLE_WALL_MARGIN
                hole_vertices_bottom[0, wall_idx] = wall_base_p0 + HOLE_WALL_MARGIN
                hole_vertices_bottom[2, wall_idx] = wall_base_p0 + HOLE_WALL_MARGIN
                asset_pos_off_p0 = wall_base_p0 + HOLE_WALL_MARGIN - hole_base_p0

            if hole_base_too_close_1:
                hole_vertices_top[1, wall_idx] = wall_base_p1 - HOLE_WALL_MARGIN
                hole_vertices_top[3, wall_idx] = wall_base_p1 - HOLE_WALL_MARGIN
                hole_vertices_bottom[1, wall_idx] = wall_base_p1 - HOLE_WALL_MARGIN
                hole_vertices_bottom[3, wall_idx] = wall_base_p1 - HOLE_WALL_MARGIN
                asset_pos_off_p1 = wall_base_p1 - HOLE_WALL_MARGIN - hole_base_p1

            if asset_position_3d is not None:
                if hole_base_too_close_0 and hole_base_too_close_1:
                    asset_position_3d[wall_idx] += (asset_pos_off_p0 + asset_pos_off_p1) / 2.0
                elif hole_base_too_close_0:
                    asset_position_3d[wall_idx] += asset_pos_off_p0
                elif hole_base_too_close_1:
                    asset_position_3d[wall_idx] += asset_pos_off_p1

    # Extrude the flat mesh with a thickness 't'
    all_vertices = np.concatenate((wall_vertices, hole_vertices_top, hole_vertices_bottom))

    wall_top_triangles = np.array(
        [
            [0, 1, 8],
            [1, 9, 8],
            [1, 3, 9],
            [3, 11, 9],
            [3, 2, 11],
            [2, 10, 11],
            [2, 0, 10],
            [0, 8, 10],
        ]
    )

    wall_bottom_triangles = np.array(wall_top_triangles) + 4
    wall_bottom_triangles = [list(reversed(triangle)) for triangle in wall_bottom_triangles]

    # TODO: Wall vs DOOR
    if type == "window":
        triangles_outer_sides = [
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 7],
            [1, 7, 3],
            [3, 7, 6],
            [3, 6, 2],
            [2, 6, 4],
            [2, 4, 0],
        ]

        triangles_inner_sides = np.array(triangles_outer_sides) + 8
        triangles_inner_sides = [list(reversed(triangle)) for triangle in triangles_inner_sides]

        # Create a new mesh for the extruded part
        all_triangles = np.concatenate(
            (
                wall_top_triangles,
                wall_bottom_triangles,
                triangles_outer_sides,
                triangles_inner_sides,
            )
        )  #
        if direction < 0:
            all_triangles = [list(reversed(triangle)) for triangle in all_triangles]

    elif type == "door":
        # 2       3
        #   10 11
        #   8 9
        # 0       1
        #    12 13
        triangles_outer_sides = [
            [0, 4, 12],
            [0, 12, 8],
            [9, 13, 5],
            [9, 5, 1],
            [1, 5, 7],
            [1, 7, 3],
            [3, 7, 6],
            [3, 6, 2],
            [2, 6, 4],
            [2, 4, 0],
        ]

        triangles_inner_sides = [
            [11, 15, 14],
            [11, 14, 10],
            [10, 14, 12],
            [10, 12, 8],
            [11, 15, 13],
            [11, 13, 9],
        ]
        triangles_inner_sides = [list(reversed(triangle)) for triangle in triangles_inner_sides]

        # Create a new mesh for the extruded part
        all_triangles = np.concatenate(
            (wall_top_triangles, wall_bottom_triangles, triangles_outer_sides)
        )  # triangles_inner_sides
        if direction < 0:
            all_triangles = [list(reversed(triangle)) for triangle in all_triangles]

    # Create a new mesh for the extruded part
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
    return mesh, asset_position_3d, asset_rotation_3d


def make_wall_colliders(wall_mesh, hole_type):
    """Assumes only rectangular holes for now. Custom function because it's simple"""
    mesh_colliders = []
    wall_vertices = wall_mesh.vertices

    # extrapolate two points
    height = wall_vertices[2][1]
    left_point = np.array([wall_vertices[10], wall_vertices[14]])  # top face and bottom face
    left_point[:, 1] = height
    right_point = np.array([wall_vertices[11], wall_vertices[15]])
    right_point[:, 1] = height

    # left of the hole
    vertices = np.array(
        [
            wall_vertices[0],
            wall_vertices[8],
            wall_vertices[2],
            left_point[0],
            wall_vertices[4],
            wall_vertices[12],
            wall_vertices[6],
            left_point[1],
        ]
    )
    triangles_face_top = [
        [0, 1, 2],
        [1, 3, 2],
    ]
    triangles_face_bottom = [
        [4, 5, 6],
        [5, 7, 6],  # top face, bottom face,
    ]
    triangles_side = [
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 7],
        [1, 7, 3],
        [3, 7, 6],
        [3, 6, 2],
        [2, 6, 4],
        [2, 4, 0],
    ]
    triangles_face_top = np.array([list(reversed(triangle)) for triangle in triangles_face_top])
    triangles_side = np.array([list(reversed(triangle)) for triangle in triangles_side])

    triangles = []
    triangles.extend(triangles_face_top)
    triangles.extend(triangles_face_bottom)
    triangles.extend(triangles_side)
    triangles = np.array(triangles)

    mesh_left = o3d.geometry.TriangleMesh()
    mesh_left.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_left.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_colliders.append(mesh_left)

    # right of the hole
    vertices = np.array(
        [
            wall_vertices[9],
            wall_vertices[1],
            right_point[0],
            wall_vertices[3],
            wall_vertices[13],
            wall_vertices[5],
            right_point[1],
            wall_vertices[7],
        ]
    )
    mesh_right = o3d.geometry.TriangleMesh()
    mesh_right.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_right.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_colliders.append(mesh_right)

    # top of the hole
    vertices = np.array(
        [
            wall_vertices[10],
            wall_vertices[11],
            left_point[0],
            right_point[0],
            wall_vertices[14],
            wall_vertices[15],
            left_point[1],
            right_point[1],
        ]
    )
    mesh_top = o3d.geometry.TriangleMesh()
    mesh_top.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_top.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_colliders.append(mesh_top)

    if hole_type == "window":
        # bottom of the hole
        left_point[:][1] = vertices[0][1]
        right_point[:][1] = vertices[0][1]

        vertices = np.array(
            [
                wall_vertices[8],
                wall_vertices[9],
                left_point[0],
                right_point[0],
                wall_vertices[12],
                wall_vertices[13],
                left_point[1],
                right_point[1],
            ]
        )
        mesh_bottom = o3d.geometry.TriangleMesh()
        mesh_bottom.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_bottom.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_colliders.append(mesh_bottom)

    return mesh_colliders


def generate_body_name(
    lemma: str,
    hash_of_type: str,
    count_of_type: int,
    body_idx: int,
    room_id: int,
) -> str:
    return f"{lemma}_{hash_of_type}_{count_of_type}_{body_idx}_{room_id}"


def generate_geom_name(
    geom: mj.MjsGeom,
    lemma: str,
    hash_of_type: str,
    count_of_type: int,
    body_idx: int,
    room_id: int,
    element_idx: int,
) -> str:
    # NOTE(wilbert): could use the original name of the geom as well if required
    geom_class = "visual" if geom.classname.name == VISUAL_CLASS else "collision"
    return f"{lemma}_{hash_of_type}_{count_of_type}_{body_idx}_{room_id}_{geom_class}_{element_idx}"


def generate_site_name(
    site: mj.MjsSite,
    lemma: str,
    hash_of_type: str,
    count_of_type: int,
    body_idx: int,
    room_id: int,
    element_idx: int,
    asset_id: str,
    object_id: str,
) -> str:
    # NOTE(wilbert): we're keeping part of the original name, so we can filter using that part
    _name = site.name if site.name != "" else "site"
    site_id = _name.replace(object_id, "").replace(asset_id, "").replace("_", "").replace("|", "")
    site_id = site_id if site_id != "" else "site"
    return f"{lemma}_{hash_of_type}_{count_of_type}_{body_idx}_{room_id}_{site_id}_{element_idx}"


def generate_joint_name(
    joint: mj.MjsJoint,
    lemma: str,
    hash_of_type: str,
    count_of_type: int,
    body_idx: int,
    room_id: int,
    element_idx: int,
    asset_id: str,
    object_id: str,
) -> str:
    _name = joint.name if joint.name != "" else "jnt"
    if joint.type == mj.mjtJoint.mjJNT_FREE:
        _name = "jntfree"
    joint_id = _name.replace(object_id, "").replace(asset_id, "").replace("_", "").replace("|", "")
    return f"{lemma}_{hash_of_type}_{count_of_type}_{body_idx}_{room_id}_{joint_id}_{element_idx}"


def change_name_recursively(
    body_spec: mj.MjsBody,
    hash_of_type: str,
    count_of_type: int,
    lemma: str,
    counter: Counter,
    asset_id: str,
    object_id: str,
    room_id: int,
    name_map: ObjectNameMapping,
) -> None:
    orig_body_name = body_spec.name
    new_body_name = generate_body_name(lemma, hash_of_type, count_of_type, counter["body"], room_id)
    name_map.bodies[new_body_name] = orig_body_name
    body_spec.name = new_body_name

    for geom in body_spec.geoms:
        assert isinstance(geom, mj.MjsGeom)
        # orig_geom_name = geom.name
        new_geom_name = generate_geom_name(
            geom=geom,
            lemma=lemma,
            hash_of_type=hash_of_type,
            count_of_type=count_of_type,
            body_idx=counter["body"],
            room_id=room_id,
            element_idx=counter["geom"],
        )
        # TODO(wilbert): commented for now, as it increases json file size by a lot
        # name_map.geoms[new_geom_name] = orig_geom_name
        geom.name = new_geom_name
        counter["geom"] += 1

    for site in body_spec.sites:
        assert isinstance(site, mj.MjsSite)
        orig_site_name = site.name
        new_site_name = generate_site_name(
            site=site,
            lemma=lemma,
            hash_of_type=hash_of_type,
            count_of_type=count_of_type,
            body_idx=counter["body"],
            room_id=room_id,
            element_idx=counter["site"],
            asset_id=asset_id,
            object_id=object_id,
        )
        site.group = DEFAULT_SITES_GROUP
        name_map.sites[new_site_name] = orig_site_name
        site.name = new_site_name
        counter["site"] += 1

    for joint in body_spec.joints:
        assert isinstance(joint, mj.MjsJoint)
        orig_joint_name = joint.name
        new_joint_name = generate_joint_name(
            joint=joint,
            lemma=lemma,
            hash_of_type=hash_of_type,
            count_of_type=count_of_type,
            body_idx=counter["body"],
            room_id=room_id,
            element_idx=counter["joint"],
            asset_id=asset_id,
            object_id=object_id,
        )
        name_map.joints[new_joint_name] = orig_joint_name
        joint.name = new_joint_name
        counter["joint"] += 1

    counter["body"] += 1
    for child_body in body_spec.bodies:
        assert isinstance(child_body, mj.MjsBody)
        change_name_recursively(
            child_body,
            hash_of_type=hash_of_type,
            count_of_type=count_of_type,
            lemma=lemma,
            counter=counter,
            asset_id=asset_id,
            object_id=object_id,
            room_id=room_id,
            name_map=name_map,
        )


def is_free_body(model: mj.MjModel, body_id: int) -> bool:
    return body_id != 0 and model.body_dofnum[body_id].item() == 6


def get_free_bodies_ids(model: mj.MjModel) -> list[int]:
    return [bid for bid in range(model.nbody) if is_free_body(model, bid)]


def get_root_bodies_handles(spec: mj.MjSpec) -> list[mj.MjsBody]:
    root_bodies_handles: list[mj.MjsBody] = []
    body_handle = spec.worldbody.first_body()
    while body_handle is not None:
        if any(joint.type == mj.mjtJoint.mjJNT_FREE for joint in body_handle.joints):
            root_bodies_handles.append(body_handle)
        body_handle = spec.worldbody.next_body(body_handle)
    return root_bodies_handles
