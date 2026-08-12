import gc
import glob
import json
import logging
import os
import re

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.linalg_utils import homogenize, inverse_homogeneous_matrix, single_or_batch
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

log = logging.getLogger(__name__)


def _get_renderer(
    model: MjModel, width: int, height: int, device_id: int, use_filament: bool = False
) -> MjOpenGLRenderer | MjFilamentRenderer:
    renderer: MjOpenGLRenderer | MjFilamentRenderer | None = None
    if use_filament:
        renderer = MjFilamentRenderer(MjModelBindings(model), height=height, width=width)
    else:
        renderer = MjOpenGLRenderer(
            MjModelBindings(model), height=height, width=width, device_id=device_id
        )
    return renderer


def _delete_blacklisted_bodies(spec: mujoco.MjSpec) -> int:
    """Delete bodies from the spec that match blacklisted asset UIDs.

    This prevents compile errors from known problematic assets like ceiling tiles
    with invalid mass/inertia. Bodies are matched if their name contains a
    blacklisted UID hash.

    Args:
        spec: The MuJoCo spec to modify in-place.

    Returns:
        Number of bodies deleted.
    """
    from molmo_spaces.tasks.task_sampler import get_static_asset_blacklist

    blacklist = get_static_asset_blacklist()
    if not blacklist:
        return 0

    # Collect bodies to delete (can't modify while iterating)
    bodies_to_delete = []

    def collect_blacklisted_bodies(body_spec: mujoco.MjsBody) -> None:
        """Recursively find bodies whose names contain a blacklisted UID."""
        body_name = body_spec.name or ""
        for uid in blacklist:
            if uid in body_name:
                bodies_to_delete.append(body_spec)
                break  # Don't check other UIDs for this body

        # Recursively check child bodies
        for child_body in body_spec.bodies:
            collect_blacklisted_bodies(child_body)

    collect_blacklisted_bodies(spec.worldbody)

    # Delete collected bodies
    for body in bodies_to_delete:
        log.debug(f"Deleting blacklisted body: {body.name}")
        spec.delete(body)

    if bodies_to_delete:
        log.info(f"Deleted {len(bodies_to_delete)} blacklisted bodies from scene")

    return len(bodies_to_delete)


def _handle_compile_error_and_blacklist(error: Exception) -> None:
    """Parse MuJoCo compile error and add problematic asset to static blacklist.

    Handles errors like:
        ValueError: Error: mass and inertia of moving bodies must be larger than mjMINVAL
        Element name 'objaceilingpanel_a3d6f7df9ff94ed59f95d5086d5f3fdd_1_0_4', id 264, line 6249
    """
    error_str = str(error)

    # Only handle mass/inertia errors (these are asset-specific and worth blacklisting)
    if "mass and inertia" not in error_str.lower():
        return

    # Extract element name from error message
    match = re.search(r"Element name '([^']+)'", error_str)
    if not match:
        return

    element_name = match.group(1)

    # Extract the MD5 hash from the element name (32 hex chars)
    hash_match = re.search(r"[a-f0-9]{32}", element_name, re.IGNORECASE)
    if not hash_match:
        return

    asset_hash = hash_match.group(0)

    # Add to static blacklist (import here to avoid circular imports)
    from molmo_spaces.tasks.task_sampler import add_to_static_blacklist

    add_to_static_blacklist(asset_hash, f"mass/inertia error from {element_name}")


def circular_kernel(radius: int):
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


class THORMap:
    """Map of the Mujoco scene.
    including fixed, hinged/articulatable, and free objects.
    exclusing dynamic agent
    """

    MAP_TYPES = ["occupancy", "voxel"]

    # 2D maps
    _occupancy_map = None
    _occupancy_scale_factor = None
    _occupancy_world_dims = None

    # 3D maps
    _voxel_map = None
    _voxel_scale_factor = None

    def __init__(
        self,
        occupancy_map=None,
        occupancy_scale_factor=None,
        occupancy_world_dims=None,
        voxel_map=None,
        voxel_scale_factor=None,
        px_per_m: int = 100,
        use_filament: bool = False,
    ):
        self._occupancy_map = occupancy_map
        self._occupancy_scale_factor = occupancy_scale_factor
        self._occupancy_world_dims = occupancy_world_dims
        self._voxel_map = voxel_map
        self._voxel_scale_factor = voxel_scale_factor
        self._px_per_m = px_per_m
        self._use_filament = use_filament

    @property
    def occupancy_map(self):
        return self._occupancy_map

    @property
    def occupancy_scale_factor(self):
        return self._occupancy_scale_factor

    @property
    def occupancy_world_dims(self):
        return self._occupancy_world_dims

    @property
    def voxel_map(self):
        return self._voxel_map

    @property
    def voxel_scale_to_world(self):
        return self._voxel_scale_to_world

    def __call__(self, r, c, map_type: str = "occupancy"):
        if map_type == "occupancy":
            position = np.zeros(3)
            h, w = self.occupancy_map.shape
            position[0] = self._occupancy_scale_factor * (c - w / 2)
            position[1] = self._occupancy_scale_factor * (r - h / 2)
            return position
        elif map_type == "voxel":
            raise NotImplementedError("Voxel map is not implemented yet")

    def _apply_buffer_with_agent_radius(
        self, occupancy_map, agent_radius, occupancy_scale_factor=None
    ):
        if occupancy_scale_factor is None:
            occupancy_scale_factor = self.occupancy_scale_factor
        assert occupancy_scale_factor is not None, "occupancy_scale_factor is required"

        r = int(agent_radius / occupancy_scale_factor) + 1
        kernel = np.ones((r, r), np.uint8)
        return cv2.dilate(occupancy_map, kernel, iterations=1)

    def _get_occupancy_from_orthoview(self, depth_pixels):
        maxrange = np.max(depth_pixels)
        gray = maxrange - depth_pixels

        # Apply threshold to get a binary image
        assert len(np.unique(gray)) > 1, "No depth values in the image"
        thres = max(
            1, np.unique(gray)[1] * 1.25
        )  # Remove floor mesh. NOTE: 1.25 is arbitrary. it's probably better to get a distribution of depth values
        _, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        return binary

    def _get_freepace_from_orthoview(self, binary_occupancy):
        binary = binary_occupancy

        mask_hull = np.zeros_like(binary, dtype=np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            all_points = np.concatenate(contours)
            hull = cv2.convexHull(all_points)
            # largest_contour = max(contours, key=cv2.contourArea)
            # hull = cv2.convexHull(largest_contour)
            cv2.drawContours(mask_hull, [hull], -1, 255, thickness=cv2.FILLED)
        freespace_mask = ~binary & mask_hull

        # clean up
        kernel = np.ones((3, 3), np.uint8)  # Define a 5x5 kernel
        m = freespace_mask
        m = cv2.erode(m, kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = (
            np.argsort(areas)[-1] + 1
        )  # Skip background label 0. Skip largest one (background)
        m = np.ones_like(binary, dtype=np.uint8) * 0  # Start with all occupied
        m[labels == largest_label] = 255
        m = cv2.dilate(m, kernel, iterations=1)

        # occupied: 255, free: 0
        freespace_mask = ~m
        return freespace_mask

    def save_map(self, path):
        if self.occupancy_map is not None:
            assert path.endswith(".png"), "Only PNG format is supported"
            cv2.imwrite(path, self.occupancy_map)
        if self.voxel_map is not None:
            assert path.endswith(".npy"), "Only NPY format is supported"
            # np.save(path, self.voxel_map)
            raise NotImplementedError("Voxel map is not implemented yet")


def sample_around_point(
    thormap: "ProcTHORMap | iTHORMap",
    point: np.ndarray,
    radius_range: tuple[float, float],
    fallback_threshold: float = 0.05,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Sample a 2D point around a given point within a given radius.
    """
    assert point.shape == (2,), "Point must be a 2D array"

    free_points = thormap.get_free_points()
    target_dist = np.linalg.norm(free_points[:, :2] - point[None], axis=1)
    # Use proper boolean indexing for array operations
    valid_mask = (target_dist > radius_range[0]) & (target_dist < radius_range[1])
    valid_points = free_points[valid_mask]
    sq_m_per_sq_px = 1 / (thormap.px_per_m**2)
    valid_neighborhood_frac = (
        len(valid_points) * sq_m_per_sq_px / (np.pi * (radius_range[1] ** 2 - radius_range[0] ** 2))
    )

    if valid_neighborhood_frac > fallback_threshold:
        for i in range(max_iter):
            # in expectation, only loop once
            batch_size = int(np.ceil(1 / valid_neighborhood_frac).item())
            theta = np.random.uniform(0, 2 * np.pi, size=batch_size)
            r = np.random.uniform(radius_range[0], radius_range[1], size=batch_size)
            sampled_points = point[None] + np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
            sampled_points_3d = np.concatenate([sampled_points, np.zeros((batch_size, 1))], axis=1)
            valid_points = thormap.check_collision(sampled_points_3d)
            if valid_points.any():
                idxs = np.where(valid_points)[0]
                log.debug(
                    f"Sampled point from map after {i + 1} iterations, {valid_neighborhood_frac=:.1%}"
                )
                return sampled_points[idxs[0]]
        log.warning(
            f"Failed to sample a point from map after {max_iter} iterations, falling back to backup. {valid_neighborhood_frac=:.1%}"
        )
    else:
        log.warning(
            f"Less than {fallback_threshold:.0%} of the sampling area is free, sampling specific pixel. This is less robust to cross-platform variation."
        )

    return valid_points[np.random.randint(len(valid_points))]


class ProcTHORMap(THORMap):
    def __init__(
        self,
        occupancy: np.ndarray,
        world_to_map: np.ndarray,
        map_to_world: np.ndarray,
        px_per_m: int,
        room_map: np.ndarray = None,
        room_ids_to_name: dict = None,
        use_filament: bool = False,
    ):
        super().__init__(occupancy_map=occupancy, px_per_m=px_per_m, use_filament=use_filament)
        self.occupancy = occupancy
        self._room_map = room_map
        self.room_ids_to_name = room_ids_to_name
        if room_ids_to_name is not None:
            self.room_names_to_id = {v: k for k, v in room_ids_to_name.items()}
        else:
            self.room_names_to_id = None
        self.world_to_map = world_to_map
        self.map_to_world = map_to_world

    @property
    def room_map(self):
        return self._room_map

    def get_free_points(self) -> np.ndarray:
        free_points_px = np.argwhere(self.occupancy)
        return self.pos_px_to_m(free_points_px)

    def get_free_points_by_room(self, room_key: str) -> np.ndarray:
        room_id = self.room_names_to_id[room_key]
        free_points_px = np.argwhere(self.occupancy)
        free_points_px = free_points_px[
            self.room_map[free_points_px[:, 0], free_points_px[:, 1]] == room_id
        ]
        return self.pos_px_to_m(free_points_px)

    @single_or_batch
    def pos_m_to_px(self, pos_m: np.ndarray) -> np.ndarray:
        assert pos_m.ndim == 2 and pos_m.shape[-1] == 3
        return np.round(homogenize(pos_m) @ self.world_to_map.T).astype(int)

    @single_or_batch
    def pos_px_to_m(self, pos_px: np.ndarray) -> np.ndarray:
        assert pos_px.ndim == 2 and pos_px.shape[-1] == 2
        return homogenize(pos_px) @ self.map_to_world.T

    @single_or_batch
    def check_collision(self, pos: np.ndarray) -> bool | np.ndarray:
        pos_px = self.pos_m_to_px(pos)
        in_range_mask = np.all((pos_px >= 0) & (pos_px < self.occupancy.shape), axis=1)
        ret = in_range_mask  # np.empty(len(pos), dtype=bool)
        ret[in_range_mask] = self.occupancy[pos_px[in_range_mask, 0], pos_px[in_range_mask, 1]]
        # ret[~in_range_mask] = True
        return ret

    def save(self, path: str):
        if path.endswith(".png"):
            # img = Image.fromarray(self.occupancy.astype(np.uint8) * 255)
            # room_map_img = Image.fromarray(self.room_map.astype(np.uint8))

            # stack the two images as channel
            img = self.occupancy.astype(np.uint8) * 255
            room_map_img = self.room_map.astype(np.uint8)
            all_img = np.stack([img, img, room_map_img], axis=2)
            all_img = Image.fromarray(all_img.astype(np.uint8))

            metadata = PngInfo()
            metadata.add_text("world_to_map", json.dumps(self.world_to_map.tolist()))
            metadata.add_text("map_to_world", json.dumps(self.map_to_world.tolist()))
            metadata.add_text("px_per_m", json.dumps(self.px_per_m))
            metadata.add_text("room_ids_to_name", json.dumps(self.room_ids_to_name))
            all_img.save(path, pnginfo=metadata)
        elif path.endswith(".npz"):
            np.savez(
                path,
                occupancy=self.occupancy,
                room_map=self.room_map,
                room_ids_to_name=self.room_ids_to_name,
                world_to_map=self.world_to_map,
                map_to_world=self.map_to_world,
                px_per_m=self.px_per_m,
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @property
    def px_per_m(self):
        return self._px_per_m

    @classmethod
    def load(cls, path: str, agent_radius: float | None = None):
        if path.endswith(".png"):
            # stacked images
            all_img = Image.open(path)
            # first channel is occupancy, second channel is room map
            img = np.array(all_img)[:, :, 0]
            room_map = np.array(all_img)[:, :, 2]

            world_to_map = np.array(json.loads(all_img.info["world_to_map"]))
            map_to_world = np.array(json.loads(all_img.info["map_to_world"]))
            px_per_m = int(np.ceil(json.loads(all_img.info["px_per_m"])))
            room_ids_to_name = json.loads(all_img.info["room_ids_to_name"])
            room_ids_to_name = {int(k): v for k, v in room_ids_to_name.items()}
            occupancy = np.array(img) > 0
            room_map = np.array(room_map)

            if agent_radius is not None:
                occupancy = ~occupancy
                rad_px = int(agent_radius * px_per_m)
                kernel = circular_kernel(rad_px)
                occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)
                room_map[occupancy] = 0
                occupancy = ~occupancy

            return cls(
                occupancy=occupancy,
                room_map=room_map,
                room_ids_to_name=room_ids_to_name,
                world_to_map=world_to_map,
                map_to_world=map_to_world,
                px_per_m=px_per_m,
            )
        elif path.endswith(".npz"):
            data = np.load(path)

            world_to_map = data["world_to_map"]
            map_to_world = data["map_to_world"]
            px_per_m = data["px_per_m"]
            room_ids_to_name = data["room_ids_to_name"]
            occupancy = data["occupancy"]
            room_map = data["occupancy"]

            if agent_radius is not None:
                occupancy = ~occupancy
                rad_px = int(agent_radius * px_per_m)
                kernel = circular_kernel(rad_px)
                occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)
                room_map[occupancy] = 0
                occupancy = ~occupancy

            return cls(
                occupancy=occupancy,
                room_map=room_map,
                room_ids_to_name=data["room_ids_to_name"],
                world_to_map=data["world_to_map"],
                map_to_world=data["map_to_world"],
                px_per_m=data["px_per_m"],
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @staticmethod
    def safe_model_data(spec, data=None):
        # Delete bodies that match blacklisted asset UIDs (prevents compile errors)
        _delete_blacklisted_bodies(spec)

        # Create new model and data
        try:
            model = spec.compile()
        except ValueError as e:
            _handle_compile_error_and_blacklist(e)
            raise
        finally:
            del spec  # Explicitly free the spec

        if data is None:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

        return model, data

    @classmethod
    def from_mj_model_path(
        cls,
        model_path: str,
        camera: str | None = None,
        agent_radius: float | None = None,
        px_per_m: int = 100,
        data: MjData | None = None,
        device_id: int = None,
        use_filament: bool = False,
    ):
        """
        Generate a ProcTHORMap from a MuJoCo model with the open door path cleared.

        This method renders occupancy maps at three camera heights:
          - 5.0 m: Base map with full wall geometry.
          - 2.5 m and 1.5 m: Lower views that capture the door opening,
            since walls might not be visible at these heights.

        It computes a door mask as the area that is occupied at 2.5 m but free
        at 1.5 m and applies that mask to the 5.0 m map. The method also computes
        the transformation matrices for mapping between world and map coordinates.

        Returns:
          ProcTHORMap: An instance with the occupancy map having the door path cleared.
        """
        # If no simulation data provided, initialize MjData and run forward
        spec = mujoco.MjSpec.from_file(model_path)

        # Recursively collect all ceiling geoms from all bodies
        ceiling_geoms = []

        def collect_ceiling_geoms_recursively(body_spec: mujoco.MjsBody) -> None:
            """Recursively traverse all bodies and collect ceiling geoms."""
            # Check geoms in current body
            for geom in body_spec.geoms:
                geom_name = geom.name
                if geom_name and "ceiling" in geom_name.lower():
                    ceiling_geoms.append(geom)

            # Recursively check child bodies
            for child_body in body_spec.bodies:
                collect_ceiling_geoms_recursively(child_body)

        # Start recursion from worldbody
        collect_ceiling_geoms_recursively(spec.worldbody)

        # Delete all collected ceiling geoms
        for geom in ceiling_geoms:
            log.debug(f"[ProcTHORMap] Deleting ceiling geom: {geom.name}")
            spec.delete(geom)  # for mujoco>3.3.5

        model, data = cls.safe_model_data(spec, data)

        # Identify floor geom indices
        floor_ids = []
        room_ids_to_name = {}
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
                floor_ids.append(geom_id)
                room_body_id = model.geom(geom_id).bodyid.item()
                room_body_name = model.body(room_body_id).name
                assert (
                    room_body_name
                    and room_body_name.startswith("world")
                    or room_body_name.startswith("room_")
                ), "Room body name must start with 'world' or 'room_'"
                room_ids_to_name[geom_id + 1] = room_body_name  # 0 is background

        assert len(floor_ids) > 0, "No floors found in the model"

        # identify opened doors body indices
        parent_to_child = {}
        parent_names = []
        for body_id in range(model.nbody):
            # body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            root_body = model.body(model.body(body_id).rootid.item())
            root_body_id = root_body.id
            root_body_name = root_body.name
            if root_body_name and (
                root_body_name.startswith("door_")
                or root_body_name.startswith("doorway_")
                or root_body_name.startswith("doorframe_")
            ):
                if root_body_id not in parent_to_child:
                    parent_to_child[root_body_id] = []
                parent_to_child[root_body_id].append(body_id)
                parent_names.append(root_body_name)
        ### NOTE MAP REQUIRE DOOR HAS JOINTS
        door_ids = []
        doorway_ids = []
        for root_body_id, children in parent_to_child.items():
            root_body_name = model.body(root_body_id).name
            for door_id in children:
                door = model.body(door_id)
                # door_name = model.body(door_id).name
                jntadr = door.jntadr.item()
                if (
                    jntadr >= 0
                    and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE
                    and model.joint(jntadr).qpos0.item() != 0.0
                ):
                    door_ids.append(door_id)
                    doorway_ids.extend(children)  # body_id)
                if jntadr < 0:
                    # door without joint are always open or closed
                    # closed doors have 3 children bodies (frame, door and handle)
                    if len(children) == 2:  # itself, and frame
                        doorway_ids.append(door_id)  # body_id)

        doorframe_geom_ids = []
        door_geom_ids = []
        for geom_id in range(model.ngeom):
            # geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = model.geom(geom_id).bodyid.item()
            parent_body_id = model.body(body_id).parentid.item()
            if body_id in door_ids or parent_body_id in door_ids:
                door_geom_ids.append(geom_id)
            parent_body_id = model.body(body_id).rootid.item()
            if parent_body_id in doorway_ids:
                doorframe_geom_ids.append(geom_id)

        # Compute axis-aligned bounding box (AABB) for floors and add a 1 m buffer per side
        aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
        aabb_size += np.array([2, 2, 0])

        # Helper function to render occupancy map at a given camera height.
        # When cam_distance == 5.0, it also returns the cam_to_world transform.
        def render_occupancy(cam_distance: float):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = aabb_center
            cam.distance = cam_distance
            cam.azimuth = 0
            cam.elevation = -90
            cam.orthographic = 1

            h = round(px_per_m * aabb_size[0])
            w = round(px_per_m * aabb_size[1])
            effective_px = h / aabb_size[0]

            renderer = _get_renderer(
                model, width=w, height=h, device_id=device_id, use_filament=use_filament
            )
            renderer.update(data, cam)
            for camera in renderer.scene.camera:
                camera: mujoco.MjvGLCamera
                camera.orthographic = 1
                camera.frustum_bottom = -aabb_size[0] / 2
                camera.frustum_top = aabb_size[0] / 2

            renderer.enable_segmentation_rendering()
            seg = renderer.render()
            seg_geom = seg[..., 0]
            # seg_body = seg[..., 2]
            cam_to_world = None
            if cam_distance == 5.0:
                # Extract camera-to-world transformation from the first camera in the scene.
                cam_to_world = np.eye(4)
                cam_to_world[:3, 3] = renderer.scene.camera[0].pos
                camera_x_ax = np.cross(
                    renderer.scene.camera[0].up, -renderer.scene.camera[0].forward
                )
                cam_to_world[:3, :3] = np.column_stack(
                    (camera_x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
                )
                assert np.allclose(cam_to_world[:3, 2], [0, 0, 1]), (
                    "Camera must be pointing straight down"
                )
            renderer.close()

            # Mark obstacles as False, free space by geom id
            occ_room_floor = np.zeros_like(seg_geom, dtype=int)
            for fid in floor_ids:
                occ_room_floor[seg_geom == fid] = fid + 1  # 0 is background
            # cv2.imwrite(f"occ_room_floor_{px_per_m}.png", occ_room_floor*20)

            # Assemble occupancy map: mark free regions as False, obstacles as True.
            occ_floor = np.ones_like(seg_geom, dtype=bool)
            for fid in floor_ids:
                occ_floor &= seg_geom != fid
            # cv2.imwrite(f"occ_floor.png", occ_floor*255)

            # mask of doors only
            occ_door = np.zeros_like(seg_geom, dtype=bool)
            for did in door_geom_ids:
                occ_door[seg_geom == did] = True
            # cv2.imwrite(f"occ_door_{px_per_m}.png", occ_door*255)

            # mask of doorframe + doors
            occ_doorframe = np.zeros_like(seg_geom, dtype=bool)
            for did in doorframe_geom_ids:
                occ_doorframe[seg_geom == did] = True
            # cv2.imwrite(f"occ_doorframe_{px_per_m}.png", occ_doorframe*255)

            # remove door from doorframe
            occ_door_path = occ_doorframe & ~occ_door
            occ_door_path = cv2.dilate(occ_door_path.astype(np.uint8), circular_kernel(15)).astype(
                bool
            )
            # cv2.imwrite(f"occ_door_path_dilated_{px_per_m}.png", occ_door_path*255)

            # remove door path from occupied map
            occ = occ_floor
            occ[occ_door_path == 1] = False
            # cv2.imwrite(f"occ_final_{px_per_m}.png", occ*255)

            if cam_distance == 5.0:
                return occ, occ_room_floor, effective_px, (h, w), cam_to_world

            return occ, occ_room_floor, effective_px, (h, w)

        ### TODO: is this treating all doors as open?
        # Might do need to render at different heights and compare
        occ_map_5, occ_room_floor_map_5, effective_px, (h, w), cam_to_world = render_occupancy(5.0)
        # cv2.imwrite("occ_map_5.png", occ_map_5*255)

        # Apply the door mask to the base map from 5.0 m: free those regions.
        occ_final = occ_map_5.copy()
        # $occ_final[door_mask] = False
        # cv2.imwrite("occ_final.png", occ_final * 255)

        occ_room_floor_final = occ_room_floor_map_5.copy()
        if agent_radius is not None:
            rad_px = int(agent_radius * effective_px)
            kernel = circular_kernel(rad_px)
            occ_final = cv2.dilate(occ_final.astype(np.uint8), kernel).astype(bool)
            occ_room_floor_final[occ_final] = 0

        # Compute transformation matrices based on the 5.0 m rendering.
        cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
        world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)

        map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
        centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
        cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
        cam_to_world_floor[2, 2] = 0
        map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered

        # Create a new ProcTHORMap instance with the door-open occupancy map.
        px_per_m = effective_px

        # Flip the occupancy map
        occ_final = ~occ_final  # so free space is True, occupied space is False

        instance = cls(
            occupancy=occ_final,
            room_map=occ_room_floor_final,
            room_ids_to_name=room_ids_to_name,
            world_to_map=world_to_map,
            map_to_world=map_to_world,
            px_per_m=px_per_m,
        )
        # Optionally, store the original (base) occupancy map for reference.
        instance.occupancy_base = occ_map_5

        # Explicitly delete temporary MuJoCo objects before garbage collection
        # These were created for map generation and are no longer needed
        del model
        del data

        # Force garbage collection to free MuJoCo objects
        gc.collect()

        return instance


class iTHORMap(ProcTHORMap):
    def __init__(
        self,
        occupancy: np.ndarray,
        world_to_map: np.ndarray,
        map_to_world: np.ndarray,
        px_per_m: int,
    ):
        super().__init__(
            occupancy=occupancy,
            world_to_map=world_to_map,
            map_to_world=map_to_world,
            px_per_m=px_per_m,
        )

    @classmethod
    def from_mj_model_path(
        cls,
        model_path,
        camera: str | None = None,
        agent_radius: float | None = None,
        px_per_m: int = 100,
        data: MjData | None = None,
        device_id: int = None,
        use_filament: bool = False,
    ):
        # We make two passes of spec/model loading:
        #  1. determine which objects are more than 1.5m above the floor
        #  2. compute the occupancy map with high objects removed

        # Pass 1.
        spec = mujoco.MjSpec.from_file(model_path)
        model, data = cls.safe_model_data(spec)

        # Find floor geoms
        floor_ids = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" in geom_name.lower():
                if model.geom(geom_id).contype == 0:  # is "__VISUAL_MJT__":
                    floor_ids.append(geom_id)
        assert len(floor_ids) > 0, "No floors found in the model"

        # Compute top of floor and height threshold (1.5m above floor)
        aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
        z_threshold = 1.5 + (aabb_center[2] + aabb_size[2] / 2)  # floor top + 1.5

        # Find top-level body names
        high_names = set()
        low_names = set()
        for geom_id in range(model.ngeom):
            aabb_center, aabb_size = geom_aabb(model, data, [geom_id], tight_mesh=False)
            if model.geom(geom_id).contype == 0:  # is "__VISUAL_MJT__":
                min_z = aabb_center[2] - aabb_size[2] / 2
                body_id = model.geom_bodyid[geom_id]
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

                # Get top-level body name
                parts = body_name.split("_")
                parts[-2] = "0"
                body_name = "_".join(parts)

                if min_z > z_threshold:
                    high_names.add(body_name)
                else:
                    low_names.add(body_name)

        # If any low-level body is low, the top-level one cannot be considered high
        high_names -= low_names

        del data, model

        # Pass 2.
        spec = mujoco.MjSpec.from_file(model_path)

        # Delete high bodies
        for body in spec.worldbody.bodies:
            body_name = body.name
            if body_name and "ceiling" in body_name.lower():
                spec.delete(body)
            elif body_name and "light" in body_name.lower():
                spec.delete(body)
            elif body_name in high_names:
                spec.delete(body)

        model, data = cls.safe_model_data(spec)

        floor_ids = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" in geom_name.lower():
                if model.geom(geom_id).contype == 0:  # is "__VISUAL_MJT__":
                    floor_ids.append(geom_id)
        assert len(floor_ids) > 0, "No floors found in the model"

        if camera is None:
            aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
            aabb_size += np.array([2, 2, 0])  # add 1m buffer to each side
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = aabb_center
            cam.distance = 5.0
            cam.azimuth = 0
            cam.elevation = -90
            cam.orthographic = 1
            h, w = round(px_per_m * aabb_size[0]), round(px_per_m * aabb_size[1])
            px_per_m = h / aabb_size[0]  # recompute to account for rounding
            renderer = _get_renderer(
                model, width=w, height=h, device_id=device_id, use_filament=use_filament
            )
            renderer.update(data, cam)
            for camera in renderer.scene.camera:
                camera: mujoco.MjvGLCamera
                camera.orthographic = 1
                camera.frustum_bottom = -aabb_size[0] / 2
                camera.frustum_top = aabb_size[0] / 2
        else:
            cam_model = model.cam(camera)
            assert model.cam_orthographic[cam_model.id], "Camera must be orthographic"
            w, h = model.cam_resolution[cam_model.id]
            px_per_m = h / cam_model.fovy.item()
            renderer = _get_renderer(
                model, width=w, height=h, device_id=device_id, use_filament=use_filament
            )
            renderer.update(data, camera)

        cam_to_world = np.eye(4)
        cam_to_world[:3, 3] = renderer.scene.camera[0].pos
        camera_x_ax = np.cross(renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
        cam_to_world[:3, :3] = np.column_stack(
            (camera_x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
        )
        assert np.allclose(cam_to_world[:3, 2], [0, 0, 1]), "Camera must be pointing straight down"

        renderer.enable_segmentation_rendering()
        seg = renderer.render()[..., 0]
        renderer.close()

        # Assemble occumancy map from segmentation
        occupancy = np.ones_like(seg, dtype=bool)
        for floor_id in floor_ids:
            occupancy &= seg != floor_id
        # Dilate to account for agent radius
        if agent_radius is not None:
            rad_px = int(agent_radius * px_per_m)
            kernel = circular_kernel(rad_px)
            occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)
        # cv2.imwrite("ithor_occupancy.png", occupancy * 255)

        # Remove small isolated islands (likely region outside of the wall)
        # Find connected components in the free space (black regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (~occupancy).astype(np.uint8), connectivity=8
        )

        # Find the largest component
        # But sometimes the largest component is outside of the wall
        """
        if num_labels > 1:  # More than just background
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
            largest_label = np.argmax(areas) + 1  # +1 because we skipped background

            # Create a mask with only the largest component
            largest_component_mask = (labels == largest_label)

            # Remove all other small components by setting them to occupied
            occupancy = occupancy | ~largest_component_mask
        """
        # cv2.imwrite("ithor_occupancy_cleaned.png", occupancy * 255)

        # transforms (x,y,z,1) in camera frame to (row, col) in the map
        cam_to_map = np.array([[0, -px_per_m, 0, h / 2], [px_per_m, 0, 0, w / 2]])
        # transforms (x,y,z,1) in world frame to (row, col) in the map
        world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)

        # converts (row, col, 1) to (x, y, 1) in camera frame, in pixels
        map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
        # transforms (x, y, 1) from pixels to to (x, y, 1) in camera frame
        centered_to_cam = np.array([[1 / px_per_m, 0, 0], [0, 1 / px_per_m, 0], [0, 0, 1]])
        # transforms (x, y, 1) in camera frame to (x, y, 0) in world frame
        cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
        cam_to_world_floor[2, 2] = 0
        # transforms (row, col, 1) in map to (x, y, 0) in world frame
        map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered

        # Flip the occupancy map. 1 is free, 0 is occupied
        occupancy = ~occupancy

        # Explicitly delete temporary MuJoCo objects before garbage collection
        # These were created for map generation and are no longer needed
        del model
        del data

        # Force garbage collection to free MuJoCo objects
        gc.collect()

        return cls(occupancy, world_to_map, map_to_world, px_per_m)

    def save(self, path: str):
        if path.endswith(".png"):
            img = Image.fromarray(self.occupancy.astype(np.uint8) * 255)

            metadata = PngInfo()
            metadata.add_text("world_to_map", json.dumps(self.world_to_map.tolist()))
            metadata.add_text("map_to_world", json.dumps(self.map_to_world.tolist()))
            metadata.add_text("px_per_m", json.dumps(self.px_per_m))
            metadata.add_text("room_ids_to_name", json.dumps(self.room_ids_to_name))
            img.save(path, pnginfo=metadata)
        elif path.endswith(".npz"):
            np.savez(
                path,
                occupancy=self.occupancy,
                world_to_map=self.world_to_map,
                map_to_world=self.map_to_world,
                px_per_m=self.px_per_m,
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @property
    def px_per_m(self):
        return self._px_per_m

    @classmethod
    def load(cls, path: str, agent_radius: float | None = None):
        if path.endswith(".png"):
            # stacked images
            img = Image.open(path)
            # first channel is occupancy, second channel is room map

            world_to_map = np.array(json.loads(img.info["world_to_map"]))
            map_to_world = np.array(json.loads(img.info["map_to_world"]))
            px_per_m = int(np.ceil(json.loads(img.info["px_per_m"])))
            occupancy = np.array(img) > 0

            if agent_radius is not None:
                occupancy = ~occupancy
                rad_px = int(agent_radius * px_per_m)
                kernel = circular_kernel(rad_px)
                occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)
                occupancy = ~occupancy

            return cls(
                occupancy=occupancy,
                world_to_map=world_to_map,
                map_to_world=map_to_world,
                px_per_m=px_per_m,
            )
        elif path.endswith(".npz"):
            data = np.load(path)
            occupancy = data["occupancy"]

            if agent_radius is not None:
                rad_px = int(agent_radius * data["px_per_m"])
                kernel = circular_kernel(rad_px)
                occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)

            return cls(
                occupancy=occupancy,
                world_to_map=data["world_to_map"],
                map_to_world=data["map_to_world"],
                px_per_m=data["px_per_m"],
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")


if __name__ == "__main__":
    import glob

    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    run_procthor_map_generation = True
    run_ithor_map_generation = True

    if run_procthor_map_generation:
        dir_path = f"{ASSETS_DIR}/scenes/procthor-10k-train"
        xmls = glob.glob(os.path.join(dir_path, "train_1.xml"))
        print(len(xmls))
        for model_path in xmls:
            if "ceiling" in model_path:
                continue
            # model_path = "debug/good_procthor/FloorPlan319_physics.xml"
            print(model_path)
            procthormap = ProcTHORMap.from_mj_model_path(
                model_path, agent_radius=None, px_per_m=200, device_id=None
            )
            procthormap.save(model_path.replace(".xml", "_map.png"))

            # test loading
            procthormap_loaded = ProcTHORMap.load(model_path.replace(".xml", "_map.png"))
            assert np.all(procthormap.occupancy == procthormap_loaded.occupancy)
            assert np.all(procthormap.room_map == procthormap_loaded.room_map)
            assert np.all(procthormap.world_to_map == procthormap_loaded.world_to_map)
            assert np.all(procthormap.map_to_world == procthormap_loaded.map_to_world)
            assert np.isclose(procthormap.px_per_m, procthormap_loaded.px_per_m)
            assert procthormap.room_ids_to_name == procthormap_loaded.room_ids_to_name

            # plot free points
            free_points = procthormap_loaded.get_free_points_by_room("room|2")
            # plot free points
            room_map = procthormap_loaded._room_map
            free_points_px = procthormap_loaded.pos_m_to_px(free_points[0])

            plt.figure()
            # filter room map to only include room 2 , keeping two dimensions
            one_room_map = room_map
            one_room_map[one_room_map != 1] = 0  # this flattens the array to one dimension

            plt.imshow(room_map)
            plt.scatter(free_points_px[1], free_points_px[0], c="red", s=1)
            plt.savefig(model_path.replace(".xml", "_free_points.png"))
            plt.close()
            # except Exception as e:
            #    print(f"Error generating map for {model_path}: {e}")

    if run_ithor_map_generation:
        dir_path = f"{ASSETS_DIR}/scenes/ithor_091625"
        xmls = glob.glob(os.path.join(dir_path, "FloorPlan1_physics_mesh.xml"))
        print(len(xmls))
        for model_path in xmls:
            # model_path = "debug/good_iTHOR/FloorPlan319_physics.xml"
            print(model_path)
            ithormap = iTHORMap.from_mj_model_path(
                model_path, agent_radius=0.25, px_per_m=200, device_id=None
            )
            # ithormap.save("debug/iTHOR/FloorPlan7_physics_map.png")

            ithormap.save(model_path.replace(".xml", "_map.png"))
            # except Exception as e:
            #    print(f"Error generating map for {model_path}: {e}")
