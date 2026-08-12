import time

import numpy as np
import rtree.index
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union
from tqdm import tqdm
import transforms3d as t3d

from utils.transformer import rotate_quat_about_world_axis


class UnStableError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class ClutteredGenerator:
    def __init__(
        self,
        minx: float = 0.0,
        miny: float = 0.0,
        maxx: float = 1.0,
        maxy: float = 1.0,
        frame: np.ndarray = None,
        global_container: Polygon | MultiPolygon | None = None,
    ):
        if global_container is None:
            self.global_container = box(minx, miny, maxx, maxy)
        else:
            self.global_container = self._normalize_region(global_container)

        self.world_z_axis = np.array([0, 0, 1], dtype=float)

        if frame is None:
            # [x, y, z, w, x, y, z]
            self.frame = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.frame = np.array(frame, dtype=float).copy().reshape(7)

        self.prohibited_area: list[tuple[str, Polygon | MultiPolygon]] = []
        self.placed_polygons: list[tuple[str, Polygon | MultiPolygon]] = []

        self.rtree_idx = rtree.index.Index()

    def _sample_rotate_angle(self, rotate_deg) -> float:
        if rotate_deg is None:
            return 0.0

        rotate_arr = np.asarray(rotate_deg, dtype=float).reshape(-1)
        if rotate_arr.size == 0:
            return 0.0
        if rotate_arr.size == 1:
            return float(np.random.uniform(-rotate_arr[0], rotate_arr[0]))
        if rotate_arr.size == 2:
            low = float(min(rotate_arr[0], rotate_arr[1]))
            high = float(max(rotate_arr[0], rotate_arr[1]))
            return float(np.random.uniform(low, high))

        raise ValueError("rotate_deg must be a scalar or a length-2 range [min_deg, max_deg]")

    def clear_models(self):
        """
        Clear only the currently placed models.
        Keep prohibited areas unchanged, and rebuild spatial index from them.
        """
        self.placed_polygons = []
        self.rtree_idx = rtree.index.Index()

        for name, polygon in self.prohibited_area:
            self.placed_polygons.append((name, polygon))
            self.rtree_idx.insert(len(self.placed_polygons) - 1, polygon.bounds)

    def reset(
        self,
        global_container: Polygon | MultiPolygon | None = None,
        frame: np.ndarray = None,
        world_z_axis: np.ndarray = None,
    ):
        """
        Reset everything except global container and frame.
        """
        self.prohibited_area = []
        self.placed_polygons = []
        if world_z_axis is not None:
            self.world_z_axis = np.array(world_z_axis, dtype=float)
        self.rtree_idx = rtree.index.Index()
        if global_container is not None:
            self.global_container = self._normalize_region(global_container)
        if frame is not None:
            self.frame = np.array(frame, dtype=float).copy().reshape(7)

    def to_world_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Transform a local pose under self.frame into world pose.
        pose: [x, y, z, qw, qx, qy, qz]
        """
        pose = np.asarray(pose, dtype=float).reshape(7)

        world_pose = np.zeros(7, dtype=float)
        world_pose[:3] = self.frame[:3] + t3d.quaternions.rotate_vector(pose[:3], self.frame[3:])
        world_pose[3:] = t3d.quaternions.qmult(pose[3:], self.frame[3:])
        return world_pose

    def _placement_to_origin(
        self,
        placement_center_pose: np.ndarray,
        placement_center_trans: np.ndarray,
    ) -> np.ndarray:
        """
        Given:
            placement_center_pose: pose of the chosen placement point in scene
            placement_center_trans: transform from origin -> placement center in object local frame

        Solve:
            origin pose in scene
        """
        placement_center_pose = np.asarray(placement_center_pose, dtype=float).reshape(7)
        placement_center_trans = np.asarray(placement_center_trans, dtype=float).reshape(7)

        origin = np.zeros(7, dtype=float)
        origin[3:] = t3d.quaternions.qmult(
            t3d.quaternions.qinverse(placement_center_trans[3:]),
            placement_center_pose[3:],
        )
        origin[:3] = placement_center_pose[:3] - t3d.quaternions.rotate_vector(placement_center_trans[:3], origin[3:])
        return origin

    def _calc_polygon(
        self,
        origin_pose: np.ndarray,
        origin_bbox_points: np.ndarray,
        margin: float = 0.0,
    ) -> tuple[Polygon, float]:
        """
        Transform object bbox vertices to world frame,
        use XY convex hull as footprint polygon,
        and return the maximum Z value.
        """
        origin_pose = np.asarray(origin_pose, dtype=float).reshape(7)
        origin_bbox_points = np.asarray(origin_bbox_points, dtype=float).reshape(-1, 3)

        rot_mat = t3d.quaternions.quat2mat(origin_pose[3:])
        bbox_points_world = origin_bbox_points @ rot_mat.T + origin_pose[:3]

        hull = MultiPoint(bbox_points_world[:, :2]).convex_hull
        polygon = hull.buffer(margin)
        z_max = bbox_points_world[:, 2].max()

        return polygon, z_max

    def _normalize_region(self, region):
        if region is None:
            return GeometryCollection()

        if region.is_empty:
            return region

        if region.geom_type == "MultiPolygon":
            parts = []
            for geom in region.geoms:
                normalized = self._normalize_region(geom)
                if not normalized.is_empty:
                    parts.append(normalized)

            if not parts:
                return MultiPolygon()

            if all(g.geom_type == "Polygon" for g in parts):
                return MultiPolygon(parts)
            return unary_union(parts)

        if region.geom_type == "GeometryCollection":
            parts = []
            for geom in region.geoms:
                normalized = self._normalize_region(geom)
                if not normalized.is_empty:
                    parts.append(normalized)

            if not parts:
                return GeometryCollection()
            return GeometryCollection(parts)

        if region.is_valid:
            return region

        if region.geom_type == "Polygon":
            coords = list(region.exterior.coords)
            unique_coords = list(dict.fromkeys(coords))

            if len(unique_coords) == 1:
                x, y = unique_coords[0]
                return Point(x, y)

            if region.area == 0:
                if len(unique_coords) == 2:
                    return LineString(unique_coords)
                return region.convex_hull

        if region.geom_type in ("LineString", "LinearRing"):
            coords = list(region.coords)
            unique_coords = list(dict.fromkeys(coords))

            if len(unique_coords) == 1:
                x, y = unique_coords[0]
                return Point(x, y)
            return LineString(unique_coords)

        return region.buffer(0)

    def get_effective_region(
        self,
        allowed_region: Polygon | MultiPolygon | None,
    ):
        """
        Effective region for current object:
            global_container ∩ allowed_region
        """
        if allowed_region is None:
            return self.global_container

        region = self._normalize_region(allowed_region)

        if region.is_empty:
            return region

        return self._normalize_region(self.global_container.intersection(region))

    def sample_point_in_region(
        self,
        region,
        max_trials: int = 200,
    ) -> tuple[float, float] | None:
        if region.is_empty:
            return None

        if isinstance(region, Point):
            return float(region.x), float(region.y)

        if isinstance(region, MultiPoint):
            points = list(region.geoms)
            if not points:
                return None
            p = points[np.random.randint(len(points))]
            return float(p.x), float(p.y)

        if isinstance(region, LineString):
            length = region.length
            if length == 0:
                x, y = region.coords[0]
                return float(x), float(y)

            distance = np.random.uniform(0, length)
            point = region.interpolate(distance)
            return float(point.x), float(point.y)

        if isinstance(region, MultiLineString):
            lines = [line for line in region.geoms if not line.is_empty]
            if not lines:
                return None

            lengths = np.array([line.length for line in lines], dtype=float)
            total_length = lengths.sum()

            if total_length == 0:
                coords = []
                for line in lines:
                    coords.extend(list(line.coords))
                if not coords:
                    return None
                x, y = coords[np.random.randint(len(coords))]
                return float(x), float(y)

            probs = lengths / total_length
            line = lines[np.random.choice(len(lines), p=probs)]

            if line.length == 0:
                x, y = line.coords[0]
                return float(x), float(y)

            distance = np.random.uniform(0, line.length)
            point = line.interpolate(distance)
            return float(point.x), float(point.y)

        if isinstance(region, Polygon):
            minx, miny, maxx, maxy = region.bounds

            if minx == maxx and miny == maxy:
                p = Point(minx, miny)
                if region.intersects(p):
                    return float(minx), float(miny)
                return None

            for _ in range(max_trials):
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                p = Point(x, y)
                if region.contains(p):
                    return float(x), float(y)

            return None

        if isinstance(region, MultiPolygon):
            polygons = [poly for poly in region.geoms if not poly.is_empty]
            if not polygons:
                return None

            areas = np.array([poly.area for poly in polygons], dtype=float)
            total_area = areas.sum()

            if total_area == 0:
                for poly in polygons:
                    sampled = self.sample_point_in_region(poly, max_trials=max_trials)
                    if sampled is not None:
                        return sampled
                return None

            probs = areas / total_area
            poly = polygons[np.random.choice(len(polygons), p=probs)]
            return self.sample_point_in_region(poly, max_trials=max_trials)

        return None

    def add_box_prohibited_area(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        name: str = None,
    ):
        polygon = box(minx, miny, maxx, maxy)
        self.add_prohibited_area(polygon, name)

    def add_circle_prohibited_area(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        name: str = None,
    ):
        polygon = Point(center_x, center_y).buffer(radius)
        self.add_prohibited_area(polygon, name)

    def add_prohibited_area(
        self,
        polygon: Polygon | MultiPolygon,
        name: str = None,
    ):
        if polygon.is_empty:
            return

        polygon = self._normalize_region(polygon)
        if polygon.is_empty:
            return

        if name is None:
            name = f"prohibited_{len(self.prohibited_area)}"

        if isinstance(polygon, MultiPolygon):
            for i, geom in enumerate(polygon.geoms):
                sub_name = f"{name}_{i}"
                self.prohibited_area.append((sub_name, geom))
                self.placed_polygons.append((sub_name, geom))
                self.rtree_idx.insert(len(self.placed_polygons) - 1, geom.bounds)
        else:
            self.prohibited_area.append((name, polygon))
            self.placed_polygons.append((name, polygon))
            self.rtree_idx.insert(len(self.placed_polygons) - 1, polygon.bounds)

    def add_polygon(
        self,
        polygon: Polygon,
        name: str,
        allowed_region: Polygon | MultiPolygon | None = None,
        check_mode: str = "bbox",
    ) -> bool:
        effective_region = self.get_effective_region(allowed_region)

        if effective_region.is_empty:
            return False

        if check_mode == "bbox":
            if not effective_region.contains(polygon):
                return False

        if check_mode != "enforce":
            candidate_ids = list(self.rtree_idx.intersection(polygon.bounds))

            for pid in candidate_ids:
                other_polygon = self.placed_polygons[pid][1]
                if polygon.intersects(other_polygon):
                    return False

        self.placed_polygons.append((name, polygon))
        self.rtree_idx.insert(len(self.placed_polygons) - 1, polygon.bounds)
        return True

    def add_model(
        self,
        placement_center_trans_list: list | np.ndarray,
        origin_bbox_points: np.ndarray,
        rotate_rand: bool = False,
        rotate_deg: float | list | np.ndarray | None = None,
        margin: float = 0.0,
        max_attempts: int = 100,
        name: str = "model",
        allowed_region: Polygon | MultiPolygon | None = None,
        sample_point_max_trials: int = 200,
        cluttered: bool = False,
        check_mode: str = "bbox",
    ) -> tuple[bool, np.ndarray | None]:
        placement_center_trans_list = np.asarray(placement_center_trans_list, dtype=float).reshape(-1, 7)
        origin_bbox_points = np.asarray(origin_bbox_points, dtype=float).reshape(-1, 3)
        effective_region = self.get_effective_region(allowed_region)
        if effective_region.is_empty:
            return False, None, None
        if rotate_deg is None:
            rotate_deg = 0.0

        for _ in range(max_attempts):
            sampled_xy = self.sample_point_in_region(
                effective_region,
                max_trials=sample_point_max_trials,
            )
            if sampled_xy is None:
                return False, None, None

            x, y = sampled_xy
            placement_center_pose = np.array(
                [x, y, 0, 1, 0, 0, 0],
                dtype=float,
            )
            idx = np.random.randint(len(placement_center_trans_list))
            placement_center_trans = placement_center_trans_list[idx]

            origin_pose = self._placement_to_origin(
                placement_center_pose,
                placement_center_trans,
            )
            if rotate_rand:
                rotate_deg_sampled = self._sample_rotate_angle(rotate_deg)
                origin_pose[-4:] = rotate_quat_about_world_axis(
                    origin_pose[-4:], self.world_z_axis, angle_deg=rotate_deg_sampled
                )
            polygon, z_max = self._calc_polygon(
                origin_pose=origin_pose,
                origin_bbox_points=origin_bbox_points,
                margin=margin,
            )
            if cluttered and (
                (origin_pose[1] + polygon.bounds[1] < 0.0 and z_max > 0.1)
                or (origin_pose[1] + polygon.bounds[1] < -0.2 and z_max > 0.05)
            ):
                continue
            if self.add_polygon(
                polygon=polygon,
                name=name,
                check_mode=check_mode,
            ):
                return True, self.to_world_pose(origin_pose), polygon

        return False, None, None

    def add_model_with_fixed_pose(
        self,
        origin_bbox_points: np.ndarray,
        z: float,
        qpos: list | np.ndarray,
        rotate_rand: bool = False,
        rotate_deg: float | list | np.ndarray | None = None,
        margin: float = 0.0,
        max_attempts: int = 100,
        name: str = "model",
        allowed_region: Polygon | MultiPolygon | None = None,
        sample_point_max_trials: int = 200,
        cluttered: bool = False,
        check_mode: str = "bbox",
    ) -> tuple[bool, np.ndarray | None]:
        origin_bbox_points = np.asarray(origin_bbox_points, dtype=float).reshape(-1, 3)
        qpos = np.asarray(qpos, dtype=float).reshape(4)

        effective_region = self.get_effective_region(allowed_region)
        if effective_region.is_empty:
            return False, None, None
        if rotate_deg is None:
            rotate_deg = 0.0

        for _ in range(max_attempts):
            sampled_xy = self.sample_point_in_region(
                effective_region,
                max_trials=sample_point_max_trials,
            )
            if sampled_xy is None:
                return False, None, None

            x, y = sampled_xy
            origin_pose = np.array(
                [x, y, z, qpos[0], qpos[1], qpos[2], qpos[3]],
                dtype=float,
            )

            if rotate_rand:
                rotate_deg_sampled = self._sample_rotate_angle(rotate_deg)
                origin_pose[-4:] = rotate_quat_about_world_axis(
                    origin_pose[-4:], self.world_z_axis, angle_deg=rotate_deg_sampled
                )

            polygon, z_max = self._calc_polygon(
                origin_pose=origin_pose,
                origin_bbox_points=origin_bbox_points,
                margin=margin,
            )

            if cluttered and (
                (origin_pose[1] + polygon.bounds[1] < 0.0 and z_max > 0.1)
                or (origin_pose[1] + polygon.bounds[1] < -0.2 and z_max > 0.05)
            ):
                continue

            if self.add_polygon(
                polygon=polygon,
                name=name,
                check_mode=check_mode,
            ):
                return True, self.to_world_pose(origin_pose), polygon

        return False, None, None

    def add_model_from_config(
        self,
        config: dict,
        place_tag: str | list[str] = None,
        rotate_rand: bool = False,
        rotate_deg: float | list | np.ndarray | None = None,
        margin: float = 0.0,
        max_attempts: int = 100,
        name: str = "model",
        allowed_region: Polygon | MultiPolygon | None = None,
        sample_point_max_trials: int = 200,
        zlim: list | np.ndarray | None = None,
        qpos: list | np.ndarray | None = None,
        cluttered: bool = False,
        check_mode: str = "bbox",
    ) -> tuple[bool, np.ndarray | None]:
        if zlim is not None:
            zmin, zmax = zlim
            if zmin > zmax:
                return False, None, None
            z = np.random.uniform(zmin, zmax)
            if qpos is None:
                qpos = [1, 0, 0, 0]

            if (
                "geometry" not in config
                or "oriented_bbox" not in config["geometry"]
                or "vertices" not in config["geometry"]["oriented_bbox"]
            ):
                return False, None, None

            origin_bbox_points = np.asarray(config["geometry"]["oriented_bbox"]["vertices"], dtype=float).reshape(-1, 3)

            return self.add_model_with_fixed_pose(
                origin_bbox_points=origin_bbox_points,
                z=z,
                qpos=qpos,
                rotate_rand=rotate_rand,
                rotate_deg=rotate_deg,
                margin=margin,
                max_attempts=max_attempts,
                name=name,
                allowed_region=allowed_region,
                sample_point_max_trials=sample_point_max_trials,
                cluttered=cluttered,
                check_mode=check_mode,
            )

        else:
            if "active" not in config or "place" not in config["active"] or len(config["active"]["place"]) == 0:
                return False, None, None

            if (
                "geometry" not in config
                or "oriented_bbox" not in config["geometry"]
                or "vertices" not in config["geometry"]["oriented_bbox"]
            ):
                return False, None, None

            origin_bbox_points = np.asarray(config["geometry"]["oriented_bbox"]["vertices"], dtype=float).reshape(-1, 3)

            all_place_tags = []
            all_place_frames = []
            for key, data in config["active"]["place"].items():
                if (
                    data.get("projection_circle", None) is not None
                    and data["projection_circle"].get("center", None) is not None
                ):
                    all_place_tags.append(key)
                    all_place_frames.append(data["projection_circle"]["center"])

            if place_tag is None:
                place_tag = all_place_tags
            elif isinstance(place_tag, str):
                place_tag = [place_tag]

            place_candidate = []
            for idx, tag in enumerate(all_place_tags):
                if tag not in place_tag:
                    continue
                place_candidate.append(all_place_frames[idx])

            if len(place_candidate) == 0:
                return False, None, None

            place_candidate = np.asarray(place_candidate, dtype=float).reshape(-1, 7)

            return self.add_model(
                placement_center_trans_list=place_candidate,
                origin_bbox_points=origin_bbox_points,
                rotate_rand=rotate_rand,
                rotate_deg=rotate_deg,
                margin=margin,
                max_attempts=max_attempts,
                name=name,
                allowed_region=allowed_region,
                sample_point_max_trials=sample_point_max_trials,
                cluttered=cluttered,
                check_mode=check_mode,
            )

    def _plot_geometry(self, ax, geom, color="blue", alpha=0.3, name=None, draw_text=True):
        if geom.is_empty:
            return

        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha)
            if draw_text and name is not None:
                ax.text(
                    geom.centroid.x,
                    geom.centroid.y,
                    name,
                    ha="center",
                    va="center",
                    fontsize=12,
                )

        elif isinstance(geom, MultiPolygon):
            for i, sub_geom in enumerate(geom.geoms):
                sub_name = f"{name}_{i}" if (name is not None and draw_text) else None
                self._plot_geometry(
                    ax,
                    sub_geom,
                    color=color,
                    alpha=alpha,
                    name=sub_name,
                    draw_text=draw_text,
                )

    def visualize(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["font.size"] = 12

        for name, polygon in self.prohibited_area:
            self._plot_geometry(ax, polygon, color="red", alpha=0.5, name=name)

        for name, polygon in self.placed_polygons:
            self._plot_geometry(ax, polygon, color="blue", alpha=0.3, name=name)

        if isinstance(self.global_container, Polygon):
            x, y = self.global_container.exterior.xy
            ax.plot(x, y, color="black", linewidth=2)
        elif isinstance(self.global_container, MultiPolygon):
            for geom in self.global_container.geoms:
                x, y = geom.exterior.xy
                ax.plot(x, y, color="black", linewidth=2)

        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.show()

    def not_empty(self):
        return len(self.placed_polygons) > 0


if __name__ == "__main__":
    left_container = box(-0.6, -0.6, -0.1, 0.4)
    right_container = box(0.1, -0.6, 0.4, 0.4)
    global_container = MultiPolygon([left_container, right_container])

    generator = ClutteredGenerator(global_container=global_container)

    generator.add_box_prohibited_area(-0.0, -0.4, 0.3, 0.1, name="forbidden_A")
    generator.add_box_prohibited_area(-0.3, -0.1, 0.1, 0.3, name="forbidden_B")

    left_region = box(-0.6, -0.6, -0.05, 0.4)
    right_region = box(0.05, -0.6, 0.4, 0.4)

    box_info = []
    start_time = time.perf_counter()

    for i in tqdm(range(500)):
        origin_extents = np.random.uniform(0.03, 0.12, 3)
        origin_min = -origin_extents / 2.0
        origin_max = origin_extents / 2.0

        origin_bbox_points = np.array(
            [
                [origin_min[0], origin_min[1], origin_min[2]],
                [origin_max[0], origin_min[1], origin_min[2]],
                [origin_max[0], origin_max[1], origin_min[2]],
                [origin_min[0], origin_max[1], origin_min[2]],
                [origin_min[0], origin_min[1], origin_max[2]],
                [origin_max[0], origin_min[1], origin_max[2]],
                [origin_max[0], origin_max[1], origin_max[2]],
                [origin_min[0], origin_max[1], origin_max[2]],
            ],
            dtype=float,
        )

        placement_center_trans = np.zeros(7, dtype=float)
        corner_idx = np.random.randint(origin_bbox_points.shape[0])
        placement_center_trans[:3] = origin_bbox_points[corner_idx]
        placement_center_trans[3:] = t3d.euler.euler2quat(
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
        )

        allowed_region = left_region if (i % 2 == 0) else right_region

        success, origin_pose, polygon = generator.add_model(
            placement_center_trans_list=placement_center_trans,
            origin_bbox_points=origin_bbox_points,
            rotate_rand=True,
            margin=0.0,
            max_attempts=50,
            name=f"model_{i}",
            allowed_region=allowed_region,
        )

        if success:
            box_info.append((f"model_{i}", origin_pose, origin_bbox_points, placement_center_trans))

    elapsed = time.perf_counter() - start_time
    print(f"Generated {len(box_info)} boxes in {elapsed:.2f} seconds")

    generator.visualize()
